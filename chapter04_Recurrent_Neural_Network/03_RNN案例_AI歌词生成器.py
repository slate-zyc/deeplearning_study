"""
案例:
    RNN案例, 基于杰伦歌词来训练模型, 用给定的起始词, 结合长度, 来进行 AI歌词生成.

实现步骤:
    1. 获取数据, 进行分词, 获取词表.
    2. 数据预处理, 构建数据集.
    3. 搭建RNN神经网络.
    4. 训练模型.
    5. 模型预测.
"""

# 导包
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os


# 1. 获取数据, 进行分词, 获取词表.
def build_vocab():
    # 1. 定义变量, 记录: 去重后所有的词, 每行文本分词结果.
    unique_words, all_words = [], []
    # 2. 遍历数据集, 获取到每行文本.
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        # 2.1 获取到每行歌词, 进行分词.
        words = jieba.lcut(line)  # ['想要', '有', '直升机', '\n']
        # 2.2 所有分词结果记录到  all_words 中.
        all_words.append(words)  # [['想要', '有', '直升机', '\n'], [第2句歌词切词], ......]
        # 2.3 遍历分词结果, 去重后, 添加到unique_words中.
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # [修改]：确保分隔符 ' ' 一定在词表中，避免后续 word_to_index[' '] 报 KeyError
    if ' ' not in unique_words:
        unique_words.append(' ')
    # 3. 统计语料中(去重后)词的数量.
    unique_word_count = len(unique_words)  # 5703个词
    # 4. 构建词表, 字典形式, key是词, value是词的索引.
    # 例如:  {'想要': 0, '有': 1, '直升机': 2, '\n': 3, ...'冠军': 5701, '要大卖': 5702}
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    # 5. 歌词文本用词表索引表示.
    corpus_idx = []
    # 6. 遍历每一行的分词结果.
    for words in all_words:
        # 6.1 定义变量, 记录: 词索引列表.
        tmp = []
        # 6.2 获取每一行的词, 并获取相应的索引.
        for word in words:
            tmp.append(word_to_index[word])
        # 6.3 在每行词之间, 添加空格隔开（是空格在词表字典中对应的索引）.
        tmp.append(word_to_index[' '])
        # 6.4 获取文档中每个词的索引, 添加到corpus_idx中.
        corpus_idx.extend(tmp)
    # 7. 返回结果: 唯一词列表(5703个词), 词表 {'想要': 0, '有': 1, ... '要大卖': 5702},  (去重后)词的数量, 歌词文本用词表索引表示.
    return unique_words, word_to_index, unique_word_count, corpus_idx


# 2. 数据预处理, 构建数据集.
# 定义数据集类, 继承 torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    # 1. 初始化词索引, 词个数等...
    def __init__(self, corpus_idx, num_chars):
        # 1.1 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 1.2 每个句子中词的个数（人为规定）.
        self.num_chars = num_chars
        # 1.3 文档数据中词的数量, 不去重，包含插入的空格.
        self.word_count = len(self.corpus_idx)
        # [修改]：原来用 word_count // num_chars，但 __getitem__ 里把 idx 当起始位置用，
        # 导致只用到文档开头很少的数据；这里改成可用起始位置的数量
        # 滑动窗口取句子
        # self.number = max(0, self.word_count - self.num_chars)
        '''
        例如这代表文档所有词（不去重，包含插入的空格）：1 2 3 4 5 6，规定句子长度 num_chars=3，则可取出以下几个句子
        输入：1 2 3    标签：2 3 4
        输入：2 3 4    标签：3 4 5
        输入：3 4 5    标签：4 5 6
        即句子数量为 6 -3 = 3
        '''
        # 滑动窗口且步长为1，样本太大了，训练太慢了，直接还是改为按块划分加快训练
        self.number = self.word_count // self.num_chars
        # 对应的__getitem__也需要修改

    # 2. 当使用 len(obj)时, 自动调用此方法.
    def __len__(self):
        # 返回句子数量
        return self.number

    # 3. 当使用 obj[index]时, 自动调用此方法.
    # def __getitem__(self, index):
    #     # index: 指的是词的索引, 并将其修正索引值 到 文档的范围里边.
    #     """
    #     还是上面的例子：
    #     start只能从开头的取到3
    #     要确保start后面至少还有num_chars，start表示索引，所以start最大取word_count-1-num_chars
    #     并且start最小不能取负数，起码在本例中不能取-1，-2，-3 以及 -7，-8，...，
    #     为了一劳永逸，直接不让start使用负索引。
    #     """
    #     # 3.1 确保索引start在合法范围内, 避免越界, start: 当前样本的起始索引.
    #     start = min(max(index, 0), self.word_count - self.num_chars - 1)
    #     # 3.2 计算当前样本的结束索引.
    #     end = start + self.num_chars
    #     # 3.3 输入值, 从文档中取出 start ~  end 的索引的词 -> 作为 x
    #     x = self.corpus_idx[start:end]      # 切片，左闭右开
    #     # 3.4 目标值, 网络理想预测结果，标签.
    #     y = self.corpus_idx[start + 1:end + 1]
    #     # 3.5 返回输入值和目标值 -> 张量形式.
    #     return torch.tensor(x), torch.tensor(y)

    def __getitem__(self, index):
        # [修改]：用 idx 直接乘以块长度，取出一整块
        start = index * self.num_chars
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        # y 往后错一位，如果到了末尾，用语料的第一个词或者直接忽略（这里简单用截断）
        y = self.corpus_idx[start + 1: end + 1]
        return torch.tensor(x), torch.tensor(y)

# 3. 搭建RNN神经网络.
class TextGenerator(nn.Module):
    # 1. 初始化方法
    def __init__(self, unique_word_count):      # unique_word_count: 去重的词的数量(5703)
        # 1.1 初始化父类的成员.
        super().__init__()
        # 1.2 初始化词嵌入层: 语料中词的数量, 词向量的维度.
        self.ebd = nn.Embedding(unique_word_count, 128)
        # 1.3 循环网络层: 词向量维度, 隐藏层维度: 256, 网络层数: 1
        self.rnn = nn.RNN(128, 256, 1)
        # 1.4 输出层(全连接层):  特征向量维度(和隐藏向量维度一致), 词表中词的个数.
        self.out = nn.Linear(256, unique_word_count)    # 词表中每个词的概率 -> 选概率最大的哪个词作为 预测结果.

    # 2. 前向传播方法
    def forward(self, inputs, hidden):
        # 2.1 初始化 词嵌入层处理.
        # embd格式: (batch句子的数量, 句子的长度, 词向量维度)
        embd = self.ebd(inputs)
        # print(f'embd.shape: {embd.shape}')

        # 2.2 rnn处理
        # rnn格式: (句子的长度, batch句子的数量, 隐藏层维度)
        output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        # 这里也可以改1.3的代码为：self.rnn = nn.RNN(128, 256, 1, batch_first=True),那么这里的代码就不需要对embd做transpose(0, 1)

        # 2.3 全连接, 输入内容必须是二维数据, 即: 词的数量 * 词的维度
        # 输入维度: (seq_len句子数量 * batch, 词向量维度256)
        # 输出维度: (seq_len句子数量 * batch, 词表中词的个数)
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 2.4 返回结果, 预测结果, 隐藏层.
        return output, hidden

    # 3. 隐藏层的初始化方法.
    def init_hidden(self, bs):      # batch_size
        # 隐藏层初始化: [网络层数, batch, 隐藏层向量维度]
        return torch.zeros(1, bs, 256)


# 4. 训练模型.
def train():
    # 1. 构建词典.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 获取数据集.
    lyrics = LyricsDataset(corpus_idx, 32)
    # 3. 初始化(神经网络)模型
    model = TextGenerator(unique_word_count)        # 预测5703个词, 每个词的概率.
    # 4. 创建数据加载器对象.
    # 参1: 数据集对象.  参2: 批次大小(每批5个句子, 每个句子32个词)  参3: 是否打乱数据.
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 6. 定义优化器.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 7. 模型训练.
    # 7.1 定义变量, 记录训练的轮数.
    epochs = 50
    # 7.2 具体的每轮训练动作.
    for epoch in range(epochs):     # epoch: 0, 1, 2, 3...9, 分别表示: 第1轮, 第2轮, ... 第10轮.
        # 7.3 定义变量记录: 本轮开始训练时间, 迭代(批次)次数, 训练总损失.
        start, iter_num, total_loss = time.time(), 0, 0.0
        # 7.4 具体的 本轮 各批次 训练动作.
        # 遍历数据集, 后台会调用 LyricsDataset#__getitem__()方法, 获取到每个样本的数据和标签,
        for x, y in lyrics_dataloader:
            # 7.5 获取隐藏层初始值.
            # hidden = model.init_hidden(5)
            # [修改]：按当前 batch 大小动态初始化 hidden，避免最后一个 batch 不满 5 时报维度不匹配
            hidden = model.init_hidden(x.size(0))
            # 7.6 模型计算.
            output, hidden = model(x, hidden)
            # 7.7 计算损失.
            # y形状为(batch, seq_len), 需要转换成一维向量->5*32=160个词的下标索引
            # output形状为: (seq_len, batch, 词向量维度)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))  # 保证展平后的索引与output能对上，不然计算的损失是错的
            loss = criterion(output, y)     # output经过激活函数后会输出所有词（5703个词）的概率，相当于是160*5703，而y展平后是160 * 1的，y的索引范围是0-5702，这正是多元交叉熵损失的原理
            # 7.8 梯度清零 + 反向传播 + 更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 7.9 累计损失 和 迭代次数.
            total_loss += loss.item()
            iter_num += 1

        # 7.10 走到这里, 说明 本轮训练结束, 打印本轮的训练信息.
        print(f'epoch: {epoch + 1}, time: {time.time() - start:.2f}s, loss: {total_loss / iter_num:.4f}')

    # 8. 走到这里, 说明多轮训练结束(模型训练结束), 保存即可.
    # [修改]：保存前确保 ./model 目录存在，否则 torch.save 会报错
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/lyrics_generator_model.pth')


# 5. 模型预测.
def evaluate(start_word, sentence_length):
    # 1. 构建词典.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 获取模型.
    model = TextGenerator(unique_word_count)
    # 3. 加载模型参数.
    model.load_state_dict(torch.load('./model/lyrics_generator_model.pth'))
    # [修改]：切换到推理模式，规范做法
    model.eval()
    # 4. 获取隐藏层初始值.
    hidden = model.init_hidden(1)
    # 5. 将输入的 开始词 转换成 索引.
    word_idx = word_to_index[start_word]
    # 6. 定义列表, 存放: 产生的词的索引.
    generate_sentence = [word_idx]  # 开始词的索引, 是列表的: 第1个值.
    # 7. 遍历句子长度, 获取到每一个词.
    for i in range(sentence_length):
        # 7.1 模型预测.
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 7.2 获取预测结果.   argmax() 从所有结果(5703个词的概率)中, 找最大值对应的索引.
        word_idx = int(torch.argmax(output))
        # 7.3 把预测结果添加到列表中.
        generate_sentence.append(word_idx)

    # 8. 将索引转成词, 并打印.
    for idx in generate_sentence:
        print(unique_words[idx], end='')
    print()  # 补一个换行，防止预测多次时和后面的内容粘连



# 6. 测试
if __name__ == '__main__':
    torch.manual_seed(123)

    # # 1. 获取数据, 进行分词, 获取词表.
    # unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # # print(f'词的数量: \n {unique_word_count}')        # 去重后, 5703个词
    # # print(f'去重后的词: \n {unique_words}')           # ['想要', '有', '直升机', '\n', '和', '你'...'冠军', '要大卖']
    # # print(f'每个词的索引: \n {word_to_index}')        # 词表: {'想要': 0, '有': 1, '直升机': 2, '\n': 3, '和': 4, '你': 5, ... '冠军': 5701, '要大卖': 5702}
    # # print(f'文档中每个词对应的索引: \n {corpus_idx}')   # [0, 1, 2, 1 3, 40, 0, 4, 5, 6, 7, 8, 3, 40, 0, 4, 5, 9, 10, 11, 3, 40,......]
    #
    # # 2. 构建数据集
    # dataset = LyricsDataset(corpus_idx, 5)    # 以 num_chars=5举例，实际训练可以调整
    # # print(f'句子数量: {len(dataset)}')
    # # # 查看下 输入值 和 目标值.
    # # x, y = dataset[0]         # dataset[0]        # dataset[1]
    # # print(f'输入值: {x}')  # [0, 1, 2, 3, 40]    [1, 2, 3, 40, 0]
    # # print(f'目标值: {y}')  # [1, 2, 3, 40, 0]    [2, 3, 40, 0, 4]
    #
    # # 3. 创建模型对象.
    # model = TextGenerator(unique_word_count)
    # # 查看参数.
    # for name, parameter in model.named_parameters():
    #     print(f'参数名称: {name}, 参数维度: {parameter.shape}')

    # 4. 训练(并保存)模型.
    # 在 train 函数中上面主函数的代码都写过一遍了
    # train()

    # 5. 测试模型.
    # 在 evaluate 函数中上面主函数的代码也都写过一遍了
    evaluate('星星', 50)
    print('-' * 30)
    evaluate('分手', 50)
    # start_word 必须在词表中，否则无法预测，直接报错