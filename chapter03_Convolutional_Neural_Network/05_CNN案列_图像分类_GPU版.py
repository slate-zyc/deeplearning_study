"""
案例:
    演示CNN的综合案例, 图像分类.
    仅将CPU训练改为GPU训练，没有做任何其他的优化

GPU 训练说明(本文件已同时兼容 CPU 和 GPU):
    1. 代码自动选择设备: 有 GPU 用 GPU, 没有就退回 CPU, 所以同一份代码两边都能跑.
    2. 核心就 3 件事: 模型 .to(DEVICE) / 每批数据 .to(DEVICE) / 统计量 .item() 取回 Python 数字.
    3. GPU 想快, 必须把 BATCH_SIZE 放大(8 -> 256), 并给 DataLoader 开 pin_memory + num_workers + persistent_workers,
       否则小批次下 GPU 大部分时间在等数据, 反而快不起来.
       训练模式时配套使用 non_blocking=True，异步拷贝
    4. 训练完在控制台看到 "使用设备: cuda" 就说明真的用上 GPU 了.
"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary


# [新增] 定义设备对象，如果检测到CUDA则使用GPU，否则退回CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8
# 注意: CPU 上用 8 也能跑, 但 GPU 上必须放大, 否则每批的"启动 kernel + 数据拷贝"开销占主导, 提速会被吃掉.
# 但 batch_size 大了之后，每轮更新次数也会变小很多，模型训练效果依旧不理想。
# 这里仅为了用GPU将CPU版本的代码跑出来，所以没有进行任何优化


def create_dataset():
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.output(x)


def train(train_dataset):
    # [修改]
    #    pin_memory=True : 把数据放进"锁页内存", 配合 non_blocking=True 让 数据->GPU 的拷贝和计算重叠.
    #    num_workers=4   : 多进程并行读数据/做 ToTensor, 避免 GPU 空等 CPU. (Windows 下必须放在 __main__ 保护里, 本文件已满足)
    #    persistent_workers : worker 只创建一次, 后面每个 epoch 复用(默认 False = 每轮重建).
    #                   ⚠️ 不开它时, 每个 epoch 结束都会销毁 worker, 下个 epoch 再重新创建:
    #                      每个 worker 都要重新 import torch, 还要把整个数据集(CIFAR10 约 150MB)pickle 一份传过去;
    #                      实测本案例光"重建 worker"就要 20~30 秒/轮, 比真正的训练计算(约 2 秒)还久.
    #                   打开后只支付一次启动成本: 100 个 epoch 从约 50 分钟降到约 5~8 分钟.
    #                   注意: 只有 num_workers > 0 时才生效.
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=4,
                            persistent_workers=True)
    # [修改] 将模型移动到指定的GPU设备上
    model = ImageModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # GPU 训练可以改回100
    epochs = 100
    for epoch_idx in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            model.train()
            # [新增] 将每批次的输入数据和标签移动到GPU设备上
            x = x.to(DEVICE, non_blocking=True)  # non_blocking=True，异步拷贝
            y = y.to(DEVICE, non_blocking=True)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # [修改] 在sum()后面加上.item()，将GPU上的张量转换为Python数字
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        print(
            f'epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc:{total_correct / total_samples:.2f}, time:{time.time() - start:.2f}s')

    torch.save(model.state_dict(), './model/image_classification_just_GPU.pth')


def evaluate(test_dataset):
    # [修改] pin_memory=True 可保留(配合 .to(DEVICE) 能省一点拷贝时间);
    #       测试阶段不设 num_workers 也行(默认 0, 单进程足够, 也方便调试).
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    # [修改] 将实例化的模型移动到GPU设备上
    model = ImageModel().to(DEVICE)
    # [修改] 在torch.load中加入map_location=DEVICE，确保权重加载到GPU上
    model.load_state_dict(torch.load('./model/image_classification_just_GPU.pth', map_location=DEVICE))
    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        model.eval()
        # [新增] 将测试集的批次数据移动到GPU设备上
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        # [修改] 同样加上.item()将结果取回CPU转为普通数字
        total_correct += (y_pred == y).sum().item()
        total_samples += len(y)

    print(f'Acc: {total_correct / total_samples:.2f}')


if __name__ == '__main__':
    # [声明] GPU训练默认模式即使设定随机种子，也无法完全复现，有很多方面的原因，但GPU有确定性模式，这里不再弄了
    torch.manual_seed(123)
    # [新增] 固定GPU的随机种子，以保证结果可复现
    torch.cuda.manual_seed_all(123)
    # print(f'使用设备: {DEVICE}')              # 看到 cuda 就说明 GPU 生效了

    train_dataset, test_dataset = create_dataset()
    # print(f'训练集: {train_dataset.data.shape}')
    # print(f'测试集: {test_dataset.data.shape}')
    # print(f'数据集类别: {train_dataset.class_to_idx}')

    # [修改] 实例化模型并移动到GPU设备上
    # 注意: 先.to(DEVICE)再 summary.
    #    torchsummary 1.5.1 的 device 参数默认是 "cuda", 它会在 GPU 上造随机输入做前向,
    #    如果模型还在 CPU 上, 会报 "Expected all tensors to be on the same device".
    model = ImageModel().to(DEVICE)
    # summary(model, (3, 32, 32), batch_size=8)

    # train(train_dataset)      # 训练一次之后就不必再训练了
    evaluate(test_dataset)