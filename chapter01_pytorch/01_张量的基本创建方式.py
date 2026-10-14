"""
案例:
    演示张量的基本创建方式.

张量(Tensor):
    PyTorch框架属于 最常用的深度学习框架, 无论是后续要学的ANN(人工神经网络), 还是CNN(卷积神经网络), RNN(循环神经网络)
    底层在处理数据时, 都是使用 张量 来处理的.

    张量 -> 存储同一类型元素的容器, 且元素值必须是 数值 才可以.

张量的基本创建方式:
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

细节:
    Tensor方式较之于 tensor方式, 可以基于 形状来直接创建张量.

张量的数据类型:
    张量中默认的数据类型是 float32(torch.FloatTensor)
    NumPy 里浮点数默认通常是 float64，也就是双精度浮点数。所以从 NumPy 转到 PyTorch 时，如果不指定 dtype，就会变成 float64。

需要你掌握的方式:
    tensor(值, 类型), 例如:
        data = np.random.randint(0, 10, size=(2, 3))
        t3 = torch.tensor(data, dtype=torch.float)
"""

# 导包
import torch
import numpy as np

# 1. 定义函数, 演示: torch.tensor 根据指定数据创建张量
def dm01():
    # 场景1: 标量 张量
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 二维列表 -> 张量.
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)

    # 场景3: numpy nd数组 -> 张量.
    # NumPy 里浮点数默认通常是 float64，也就是双精度浮点数。所以从 NumPy 转到 PyTorch 时，如果不指定 dtype，就会变成 float64。
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data, dtype=torch.float)      # dtype 指定数据类型
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    # 场景4: 尝试直接创建 指定维度(例如: 2行3列的)张量
    # t4 = torch.tensor(2, 3)               # 报错.需要使用 Tensor
    # print(f't4: {t4}, type: {type(t4)}')
    print('=' * 30)

# 2. 定义函数, 演示: torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
def dm02():
    # 场景1: 标量 张量
    t1 = torch.Tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 二维列表 -> 张量.
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)

    # 场景3: numpy nd数组 -> 张量.
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)     # 不能指定类型
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    # 场景4: 尝试直接创建 指定维度(例如: 2行3列的)张量
    t4 = torch.Tensor(2, 3)
    print(f't4: {t4}, type: {type(t4)}')        # 生成的数是随机的
    print('=' * 30)

# 3. 定义函数, 演示: torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
def dm03():
    # 场景1: 标量 张量
    t1 = torch.IntTensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 二维列表 -> 张量.
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)

    # 场景3: numpy nd数组 -> 张量.
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    # 场景4: 如果类型不匹配, 会尝试自动转换类型.
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)        # 默认: float32，所以输出结果不显示 dtype
    print(f't4: {t4}, type: {type(t4)}')
    print('=' * 30)

# 4. 定义测试函数.
if __name__ == '__main__':
    dm01()        # 掌握
    dm02()
    dm03()