"""
案例:
    演示近30天, 天气分布情况.

结论:
    针对于 β(调节权重系数)来讲, 其值越大说明: 越依赖指数加权平均, 越不依赖本次的梯度值, 数据就越: 平缓.
"""


import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30


# 1. 实际平均温度
def dm01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10     # 乘10让数据放大10倍
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def dm02(beta=0.9):
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
    print(temperature)

    exp_weight_avg = []
    # idx从1开始
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        # idx-2：2-2=0，exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='r')
    plt.scatter(days, temperature)
    plt.show()


if __name__ == '__main__':
    dm01()      # 不考虑 权重系数, 每个值的权重都一致.
    dm02(0.5)   # 考虑 权重系数, 值越小, 数据越陡.
    dm02(0.9)   # 考虑 权重系数, 值越大, 数据越平缓.

# ------
# # 合并三个图形
# import torch
# import matplotlib.pyplot as plt
#
# # 核心修复：设置中文字体，解决中文显示不出来的问题
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号 '-' 显示为方块的问题
#
# ELEMENT_NUMBER = 30
#
#
# # 1. 实际平均温度
# def dm01(ax):
#     torch.manual_seed(0)
#     temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
#     days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
#
#     ax.plot(days, temperature, color='r')
#     ax.scatter(days, temperature, color='#1f77b4', s=35, alpha=0.8, edgecolors='white', linewidths=0.5)
#     ax.set_title("实际温度分布 (无平滑)")
#
#
# # 2. 指数加权平均温度
# def dm02(beta=0.9, ax=None):
#     torch.manual_seed(0)
#     temperature = torch.randn(size=[ELEMENT_NUMBER, ]) * 10
#
#     exp_weight_avg = []
#     for idx, temp in enumerate(temperature, 1):
#         if idx == 1:
#             exp_weight_avg.append(temp)
#             continue
#         new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
#         exp_weight_avg.append(new_temp)
#
#     days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
#
#     ax.plot(days, exp_weight_avg, color='r')
#     ax.scatter(days, temperature, color='#1f77b4', s=35, alpha=0.8, edgecolors='white', linewidths=0.5)
#     ax.set_title(f"β={beta}")
#
#
# if __name__ == '__main__':
#     # 创建 1行3列 的画布
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
#     # 绘制三个子图
#     dm01(axes[0])
#     dm02(0.5, axes[1])
#     dm02(0.9, axes[2])
#
#     plt.tight_layout()  # 自动调整间距
#     plt.show()