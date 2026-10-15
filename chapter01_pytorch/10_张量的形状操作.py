"""
案例:
    演示张量的形状操作.

涉及到的API:
    reshape():      在不改变张量内容（数据拉平成一维后要保持一致）的前提下, 对其形状做改变.
    unsqueeze()     在指定的轴上增加一个(1)维度, 等价于: 升维.
    squeeze()       删除所有为1的维度, 等价于: 降维.
    transpose()     一次只能交换2个维度.
    permute()       一次可以同时交换多个维度.
    view()          只能修改连续的张量的形状,  连续张量 = 内存中存储顺序 和 在张量中显示的顺序相同.
    contiguous()    把不连续的张量 -> 连续的张量, 即: 基于张量中显示的顺序, 修改内存中的存储顺序.
    is_contiguous() 判断张量是否是连续的.

shape 属性或者 size() 方法都可以获得张量的形状。以下是使用操作对比:
    .shape   .shape[0]  .shape[1]   .shape[-1]
    .size()  .size(0)   .size(1)    .size(-1)

细节：
    transpose 是转置的意思，所以 transpose() 和 reshape() 是有区别的，前者是多维广义转置，后者是拼接
    permute 是排列的意思，permute 的实现效果也是多维广义转置，eg: (i, j, k) -> (k, i, j)
    view() 与 reshape() 的操作结果是一样的，只是 view() 多了一个判断张量是否连续的机制
    综上，reshape() 和 view() 得到的张量是连续的，transpose() 和 permute()得到的张量是不连续的

需要你掌握的函数:
    .reshape(), .unsqueeze(), .permute(), .view()
"""

# 导包
import torch

# 指定随机种子
torch.manual_seed(24)

# 1. 定义函数, 演示: reshape()函数.
# reshape 是把张量拉平成一维，再按行优先重新切分新形状
def dm01():
    # 1. 定义2行3列的张量.
    t1 = torch.randint(1, 10, size=(2, 3))
    # 使用 shape 属性或者 size() 方法都可以获得张量的形状
    print(f't1: {t1}, shape: {t1.shape}, row: {t1.shape[0]}, columns: {t1.shape[1]}, {t1.shape[-1]}')
    print(f't1: {t1}, size: {t1.size()}, row: {t1.size(0)}, columns: {t1.size(1)}, {t1.size(-1)}')

    # 2. 通过reshape()函数, 把t1 -> 3行2列, 1行6列, 6行1列.
    t2 = t1.reshape(3, 2)
    # t2 = t1.reshape(1, 6)
    # t2 = t1.reshape(6, 1)
    print(f't2: {t2}, shape: {t2.shape}, row: {t2.shape[0]}, columns: {t2.shape[1]}, {t2.shape[-1]}')
    # print(f't1: {t1}, shape: {t1.shape}, row: {t1.shape[0]}, columns: {t1.shape[1]}, {t1.shape[-1]}')
    # 说明不是原地修改

    # 3. 尝试通过reshape()函数, 把t1 -> 2行5列的结果.
    # t3 = t1.reshape(2, 5)       # 报错, 转之前共计 2*3=6个元素, 转之后共计 2*5=10个元素, 不一致.
    # print(f't3: {t3}')
    print('*'* 30)


# 2. 定义函数, 演示: unsqueeze()函数, squeeze()函数.
def dm02():
    # 1. 定义2行3列的张量.
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}')       # (2, 3)
    print('-' * 30)

    # 2. 在0维上, 添加一个维度.
    t2 = t1.unsqueeze(0)
    print(f't2: {t2}, shape: {t2.shape}')       # (1, 2, 3)
    # print(f't1: {t1}, shape: {t1.shape}')       # (2, 3)
    # 说明不是原地修改

    # 3. 在1维上, 添加一个维度.
    t3 = t1.unsqueeze(1)
    print(f't3: {t3}, shape: {t3.shape}')       # (2, 1, 3)

    # 4. 在2维上, 添加一个维度.
    # t4 = t1.unsqueeze(2)
    t4 = t1.unsqueeze(-1)   # 效果同上
    print(f't4: {t4}, shape: {t4.shape}')       # (2, 3, 1)

    # 5. 在3微上(不存在), 添加一个维度.
    # t5 = t1.unsqueeze(3)        # 报错, 越界.
    # print(f't5: {t5}, shape: {t5.shape}')       # (2, 3, ???, 1)

    # 6. 删除所有为1的维度.
    t6 = torch.randint(1, 10, size=(2, 1, 3, 1, 1))
    print(f't6: {t6}, shape: {t6.shape}')        # (2, 1, 3, 1, 1)
    print('-' * 30)

    # dim 可以传单个数字，也可以传元组, 表示删除指定维度上大小为1的轴，如果该维度 size≠1，则不动
    # 不传 dim 表示删除所有大小为1的维度
    t7 = t6.squeeze()
    print(f't7: {t7}, shape: {t7.shape}')        # (2, 3)
    t8 = t6.squeeze(dim=1)
    print(f't8: {t8}, shape: {t8.shape}')  # (2, 3, 1, 1)
    t9 = t6.squeeze(dim=(1, 3, 4))
    print(f't9: {t9}, shape: {t9.shape}')  # (2, 3)
    print('*' * 30)


# 3. 定义函数, 演示: transpose()函数, permute()函数.
def dm03():
    # 1. 定义张量.
    t1 = torch.randint(1, 10, size=(2, 3, 4))
    print(f't1: {t1}, shape: {t1.shape}')
    print('-' * 30)

    # 2. 改变维度从 (2, 3, 4) -> (4, 3, 2)
    t2 = t1.transpose(0, 2)
    # t2 = t1.transpose(2, 0)         # 效果同上. t1.transpose(-1,-2) 交换最后两维
    # t2 = t1.transpose(0, -1)        # 效果同上.
    # t2 = torch.transpose(t1, 0, 2)  # 效果同上
    print(f't2: {t2}, shape: {t2.shape}')
    # print(f't1: {t1}, shape: {t1.shape}')
    # 说明不是原地修改

    # 3. 改变维度从 (2, 3, 4) -> (4, 2, 3)
    t3 = t1.permute(2, 0, 1)
    # t3 = t1.transpose(0, 1).transpose(0, 2)   # 效果同上
    print(f't3: {t3}, shape: {t3.shape}')
    print('*' * 30)


# 4. 定义函数, 演示: view()函数, contiguous()函数, is_contiguous()函数.
def dm04():
    # 思路: 演示 view()无法改变 不连续的张量的形状, 可以通过 is_contiguous()判断张量是否连续, 也可以通过 contiguous()把不连续的张量 -> 连续张量.
    # 1. 定义张量.
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}')

    # 2. 判断张量是否连续. 即: 张量中的顺序 和 内存中存储顺序是否一致.
    # print(t1.is_contiguous())       # True

    # 3. 通过 view()函数, 修改上述张量的形状. 从(2, 3) -> (3, 2)
    t2 = t1.view(3, 2)
    print(f't2: {t2}, shape: {t2.shape}')
    print(t2.is_contiguous())        # True

    # 4. 通过 transpose()交换维度 -> 交换之后, 不连续了.
    t3 = t1.transpose(0, 1)
    print(f't3: {t3}, shape: {t3.shape}')
    print(t3.is_contiguous())        # False

    # 5. 尝试把 t3张量 从(3, 2), 通过 view() 转成 (2, 3).
    # t4 = t3.view(2, 3)      # t3不连续, 所以view()无法改变形状. 报错.
    # print(f't4: {t4}, shape: {t4.shape}')

    # 6. 可以通过 contiguous()函数, 把 t3张量 -> 连续张量 -> 然后就能通过view修改形状了.
    t5 = t3.contiguous().view(2, 3)
    print(f't5: {t5}, shape: {t5.shape}')


# 0. reshape(), view(), transpose(), permute() 形状操作的区别
# transpose 是转置的意思，所以 transpose() 和 reshape() 是有区别的，前者是多维广义转置，后者是拼接
# permute 是排列的意思，permute 的实现效果也是多维广义转置，eg: (i, j, k) -> (k, i, j)
# view() 与 reshape() 的操作结果是一样的，只是 view() 多了一个判断张量是否连续的机制
# 综上，reshape() 和 view() 得到的张量是连续的，transpose() 和 permute()得到的张量是不连续的
def dm00():
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}')

    t2 = t1.reshape(3, 2)
    print(f't2: {t2}, shape: {t2.shape}')
    t3 = t1.transpose(0, 1)     # 相当于矩阵转置
    print(f't3: {t3}, shape: {t3.shape}')
    t4 = t1.permute(1, 0)               # 同 permute，也相当于矩阵转置
    print(f't4: {t4}, shape: {t4.shape}')

# 5. 测试
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm00()

