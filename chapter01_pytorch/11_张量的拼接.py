"""
案例:
    演示张量的拼接操作.

涉及到的API:
    cat()     不改变维度数, 拼接张量. 除了拼接的那个维度外, 其它维度数必须保持一致.
    stack()   会改变维度数, 拼接张量. 所有的维度都必须保持一致.
"""

# 导包
import torch

# 设置随机种子.
torch.manual_seed(24)

# 1. 创建两个张量.
t1 = torch.randint(1, 10, (2, 3))
print(f't1: {t1}, shape: {t1.shape}')

t2 = torch.randint(1, 10, (2, 3))
print(f't2: {t2}, shape: {t2.shape}')
'''
设置随机种子后，每次运行程序，得到的 t1 是固定的，t2 也是固定的。
也就是说，第一次运行和第二次运行会得到完全相同的 t1 和 t2。
但 t1 和 t2 不是同一个结果，因为：
    t1 消耗了随机序列的第一组数；
    t2 接着消耗了随机序列的第二组数。
'''

# 2. 演示张量的拼接
# 思路1: cat() 拼接张量.
t3 = torch.cat([t1, t2], dim=0)     # (2, 3) + (2, 3) = (4, 3)
print(f't3: {t3}, shape: {t3.shape}')

t4 = torch.cat([t1, t2], dim=1)     # (2, 3) + (2, 3) = (2, 6)
print(f't4: {t4}, shape: {t4.shape}')

t5 = torch.cat([t1, t2], dim=-1)    # 效果同上
print(f't5: {t5}, shape: {t5.shape}')

# t6 = torch.cat([t1, t2], dim=2) # 报错, 越界
# print(f't6: {t6}, shape: {t6.shape}')
print('-' * 30)

# 思路2: stack() 拼接张量, 可以是新维度, 但是无论新旧维度, 所有维度都必须保持一致.
t7 = torch.stack([t1, t2], dim=0)   # (2, 3) + (2, 3) = (2, 2, 3)
print(f't7: {t7}, shape: {t7.shape}')      # 相当于再套一层括号

t8 = torch.stack([t1, t2], dim=1)   # (2, 3) + (2, 3) = (2, 2, 3)
print(f't8: {t8}, shape: {t8.shape}')      # 一维度对应拼接

t9 = torch.stack([t1, t2], dim=2)   # (2, 3) + (2, 3) = (2, 3, 2)
print(f't9: {t9}, shape: {t9.shape}')      # 二维度对应拼接，实现效果相当于上面情况的"广义转置"

# 报错, 越界
# t10 = torch.stack([t1, t2], dim=3)   # (2, 3) + (2, 3) = (2, 3, ?, 2)
# print(f't10: {t10}, shape: {t10.shape}')
