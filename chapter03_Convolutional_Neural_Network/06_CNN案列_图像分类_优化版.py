"""
案例:
    演示CNN的综合案例, 图像分类.
    优化且使用GPU大幅加速训练, 并重点解决过拟合问题.

GPU 训练说明(本文件已同时兼容 CPU 和 GPU):
    1. 代码自动选择设备: 有 GPU 用 GPU, 没有就退回 CPU, 所以同一份代码两边都能跑.
    2. 核心就 3 件事: 模型 .to(DEVICE) / 每批数据 .to(DEVICE) / 统计量 .item() 取回 Python 数字.
    3. GPU 想快, 必须把 BATCH_SIZE 放大(8 -> 128/256), 并给 DataLoader 开 pin_memory + num_workers + persistent_workers,
       否则小批次下 GPU 大部分时间在等数据, 反而快不起来.
       训练模式时配套使用 non_blocking=True，异步拷贝
    4. 训练完在控制台看到 "使用设备: cuda" 就说明真的用上 GPU 了.

优化方向:
    1. 数据增强: RandomCrop + RandomHorizontalFlip, 相当于"免费扩容"训练集.
    2. 归一化: Normalize 用 CIFAR10 统计量, 让输入分布更规整, 训练更稳更快.
    3. 增大参数量且分配合适的结构，避免"卷积弱 + FC 胖" ，这样容易天然容易死记硬背，造成过拟合
        卷积加强(3->64→128→256 + BatchNorm), FC 4096->512->256->10.
    4. 在两个全连接层后使用 dropout 正则化.
    4. 优化器: Adam -> AdamW + weight_decay(解耦的 L2 正则).
    5. 学习率调度: CosineAnnealingLR, 后期自动降 lr, 避免在最优解附近震荡记噪声.
    6. 早停: 验证集连续多轮不提升就停, 防止练过头.
    7. 增加训练轮数，之前受限于 CPU 训练速度，训练轮数太少.
    7. 划分验证集: 从训练集里划 10%, 调参/早停用 val, 测试集只在最后评一次.
"""

# 导包
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

# 设备选择: 有 GPU 用 GPU, 没有则自动退回 CPU(同一份代码两边都能跑)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 上必须放大 batch, 否则每批的 kernel 启动 + 数据拷贝开销占主导, 提速会被吃掉
BATCH_SIZE = 128


# 1. 准备数据集.
def create_dataset():
    # todo 归一化统计量用 CIFAR10 官方给的 RGB 均值/标准差, 训练集和测试集必须用同一组.
    #      不能拿测试集自己算, 否则等于泄漏了测试集信息.
    norm = Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    # todo 训练集做随机增强: 每次取图都略有不同, 等价于扩充了数据量, 是最有效的抗过拟合手段.
    #      RandomCrop(32, padding=4): 四周补 4 像素再随机裁回 32, 让模型不依赖物体绝对位置.
    #      RandomHorizontalFlip: 随机水平翻转, CIFAR10 里猫/车翻转后语义不变, 可以放心用.
    train_tf = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        norm,
    ])
    # todo 测试集绝不能用随机变换, 否则每次评估结果都在抖, 无法比较.
    test_tf = Compose([ToTensor(), norm])

    train_full = CIFAR10(root='./data', train=True, transform=train_tf, download=True)
    test_set = CIFAR10(root='./data', train=False, transform=test_tf, download=True)

    # todo 从训练集划 10% 做验证集: 早停/调参看 val, 测试集只在最后用一次, 评估更客观.
    n_val = 5000
    train_set, val_set = random_split(train_full, [len(train_full) - n_val, n_val])
    return train_set, val_set, test_set


# 2. 搭建(卷积)神经网络: 加强卷积、收敛 FC 容量.
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # todo padding=1 保证卷积后尺寸不变(HW 不变), 尺寸只由 pool 决定, 便于推算 FC 输入.
        #      原模型不加 padding, 每层卷积都会缩尺寸, 特征图越来越小.
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # todo 多一层卷积: 更深的卷积能提取更抽象的特征, 把"记忆"的活从 FC 层挪回卷积层.
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 尺寸: 32 -> 16 -> 8 -> 4

        # todo FC 4096->512->256->10.
        #      输入维度 = 通道 256 * 4 * 4 = 4096, 由上面三次 pool 精确推得.
        self.linear1 = nn.Linear(256 * 4 * 4, 512)
        self.linear2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 10)

        # Dropout 只加在全连接层, 卷积层后不加(卷积参数少, 加了容易欠拟合)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # todo 顺序惯例: Conv -> BN -> ReLU -> Pool.
        #      BN 放在 ReLU 前, 让每层输入分布稳定, 训练更快也更稳, 附带一点正则效果.
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))   # (B, 64, 16, 16)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))   # (B, 128, 8, 8)
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))   # (B, 256, 4, 4)

        # 全连接层只能处理二维数据, 拉平成 (B, 4096)
        x = x.reshape(x.size(0), -1)

        # Dropout 只在 model.train() 时生效, model.eval() 时自动关闭
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.dropout(torch.relu(self.linear2(x)))
        # 输出层前不加 dropout, 交给 CrossEntropyLoss 里的 softmax 处理
        return self.output(x)


# 3. 通用评估函数(训练中评估验证集 / 最后评估测试集都用它).
@torch.no_grad()   # todo 评估不需要梯度, 关掉能省显存、提速, 也避免误改参数
def evaluate(model, loader):
    model.eval()   # 切评估模式: 关闭 Dropout, BN 用 running 统计量
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        correct += (model(x).argmax(dim=-1) == y).sum().item()
        total += len(y)
    return correct / total


# 4. 模型训练.
def train(train_set, val_set):
    # todo pin_memory: 把数据放进锁页内存, 配合 non_blocking=True 让拷贝和计算重叠.
    #      num_workers=4: 多进程预取数据, 避免 GPU 空等 CPU.
    #      persistent_workers: worker 只建一次, 每轮复用. 不开的话每轮重建要 20~30 秒, 比训练本身还久.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=4, persistent_workers=True)
    # 验证集不用 shuffle, 评估结果才稳定
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=4, persistent_workers=True)

    model = ImageModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵 = softmax + 损失计算

    # todo Adam -> AdamW: Adam 的 weight_decay 和自适应学习率耦合得不好, AdamW 是解耦版本, 正则更有效.
    #      weight_decay=5e-4: 惩罚过大的权重, 抑制模型去拟合噪声.
    #      lr不变: 加了 BN 后对学习率不敏感, 太小的 lr 配 100 轮会练得太慢.
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    epochs = 100
    # todo 余弦退火: lr 从 1e-3 平滑降到接近 0, 后期步长变小, 不会在最优点附近震荡记噪声.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # todo 早停: val 连续 patience 轮不提升就停, 并保存最佳权重, 防止练过头.
    best_acc, patience, wait = 0.0, 10, 0

    for epoch in range(epochs):
        model.train()   # 切训练模式: 开启 Dropout, BN 用当前批次统计量
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        for x, y in train_loader:
            # 每批数据和模型必须在同一设备, 否则报 "Expected all tensors to be on the same device"
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # .item() 把 GPU 上的 0 维张量取回成 Python 数字, 避免累加器一直是 GPU 张量
            total_correct += (y_pred.argmax(dim=-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_acc = total_correct / total_samples
        # todo 每轮评估验证集: 只有同时看 train_acc 和 val_acc 的 gap, 才知道过拟合从第几轮开始
        val_acc = evaluate(model, val_loader)
        scheduler.step()   # 每个 epoch 结束后更新学习率

        print(f'epoch {epoch + 1:3d} | loss {total_loss / total_samples:.4f} | '
              f'train_acc {train_acc:.3f} | val_acc {val_acc:.3f} | '
              f'gap_acc {train_acc - val_acc:.3f} | {time.time() - start:.1f}s')

        # 保存验证集上表现最好的一轮, 最后测试用这个权重
        if val_acc > best_acc:
            best_acc, wait = val_acc, 0
            torch.save(model.state_dict(), './model/image_classification_best.pth')
        else:
            wait += 1
            if wait >= patience:
                print(f'早停 @ epoch {epoch + 1}, best val_acc = {best_acc:.3f}')
                break


# 5. 测试
if __name__ == '__main__':
    # GPU 训练默认模式即使固定种子也无法完全复现(并行归约顺序不确定), 这里只固定种子不追求严格复现
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    print(f'使用设备: {DEVICE}')   # 看到 cuda 就说明 GPU 生效了

    # 1. 获取数据集: 训练集(带增强) / 验证集 / 测试集
    train_set, val_set, test_set = create_dataset()

    # 2. 查看模型结构. 先 .to(DEVICE) 再 summary, 否则 torchsummary 在 GPU 上造输入会因设备不一致报错
    model = ImageModel().to(DEVICE)
    summary(model, (3, 32, 32), batch_size=BATCH_SIZE)

    # 3. 模型训练(内部带早停, 自动保存最佳权重)
    train(train_set, val_set)

    # 4. 模型测试: 用验证集上最好的那轮权重, 在测试集上评一次
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    best_model = ImageModel().to(DEVICE)
    # map_location=DEVICE: 明确权重加载到哪个设备, 避免 GPU/CPU 权重跨设备加载报错
    best_model.load_state_dict(torch.load('./model/image_classification_best.pth', map_location=DEVICE))
    print(f'最终 test_acc = {evaluate(best_model, test_loader):.4f}')