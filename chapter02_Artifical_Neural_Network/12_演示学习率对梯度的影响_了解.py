"""
案例:
    通过代码演示学习率 对 梯度(权重优化)的影响.

结论:
    学习率越小, 梯度下降越慢.
    学习率越大, 梯度下降越快, 可能会越过最小值, 造成震荡, 甚至不收敛(梯度爆炸)
"""

import torch
import matplotlib.pyplot as plt

# x看成是权重，y看成是loss，下面通过代码来理解学习率的作用
def func(x_t):
    return torch.pow(2*x_t, 2)  # y = 4 x ^2

# 采用较小的学习率，梯度下降的速度慢
# 采用较大的学习率，梯度下降太快越过了最小值点，导致不收敛，甚至震荡
def dm01():

    x = torch.tensor([2.], requires_grad=True)
    # 记录loss迭代次数，画曲线
    iter_rec, loss_rec, x_rec = list(), list(), list()

    # 实验学习率： 0.01 0.02 0.03 0.1 0.2 0.3 0.4
    # lr = 0.01     # 较小的学习率,  会导致梯度下降速度慢
    # lr = 0.1    # 正常的梯度下降
    # lr = 0.125      # 当学习率设置0.125 一下子求出一个最优解
                    # x=0 y=0 在x=0处梯度等于0 x的值x=x-lr*x.grad就不用更新了
                    # 后续再多少次迭代 都固定在最优点

    # lr = 0.2      # x从2.0一下子跨过0点，到了左侧负数区域
    lr = 0.3      # 梯度越来越大 梯度爆炸
    max_iteration = 4
    for i in range(max_iteration):
        y = func(x)   # 得出loss值
        y.backward()  # 计算x的梯度
        print("Iter:{}, X:{:8}, X.grad:{:8}, loss:{:10}".format(
            i, x.detach().numpy()[0], x.grad.detach().numpy()[0], y.item()))
        x_rec.append(x.item())      # 梯度下降点 列表
        # 更新参数
        x.data.sub_(lr * x.grad)    # x = x - x.grad
        x.grad.zero_()
        iter_rec.append(i)          # 迭代次数 列表
        loss_rec.append(y.item())  # 损失值 列表，这里将y改为y.item()以获取标量值
    # 迭代次数-损失值 关系图
    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.grid()
    plt.xlabel("Iteration X")
    plt.ylabel("Loss value Y")
    # 函数曲线-下降轨迹 显示图
    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.detach().numpy(), y.detach().numpy(), label="y = 4*x^2")
    y_rec = [func(torch.tensor(i)).item() for i in x_rec]
    print('x_rec--->', x_rec)
    print('y_rec--->', y_rec)
    # 指定线的颜色和样式（-ro：红色圆圈，b-：蓝色实线等）
    plt.subplot(122).plot(x_rec, y_rec, '-ro')
    plt.grid()
    plt.legend()
    plt.show()

dm01()


# # 合并绘图
#
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
#
# # 设置中文字体，防止中文显示为方块 (Windows一般用SimHei，Mac用Arial Unicode MS)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def func(x_t):
#     return torch.pow(2 * x_t, 2)  # y = 4 x ^2
#
#
# def run_experiment(lr, max_iteration, ax1, ax2, title_text):
#     """
#     运行单次梯度下降实验并绘图
#     """
#     x = torch.tensor([2.], requires_grad=True)
#     iter_rec, loss_rec, x_rec = list(), list(), list()
#
#     for i in range(max_iteration):
#         y = func(x)
#         y.backward()
#
#         x_rec.append(x.item())
#         loss_rec.append(y.item())
#         iter_rec.append(i)
#
#         # 更新参数: x = x - lr * grad
#         x.data.sub_(lr * x.grad)
#         x.grad.zero_()
#
#     # 绘制左图：迭代次数 vs Loss
#     ax1.plot(iter_rec, loss_rec, '-ro')
#     ax1.set_title(title_text, color='red', fontsize=14, loc='right')
#     ax1.grid()
#     ax1.set_xlabel("Iteration X")
#     ax1.set_ylabel("Loss value Y")
#     # 限制y轴范围，防止爆炸时图像压缩太严重
#     # if lr >= 0.3:
#     #     ax1.set_ylim([0, 130])
#     # else:
#     #     ax1.set_ylim([0, 18])
#
#     # 绘制右图：函数曲线和下降轨迹
#     x_t = torch.linspace(-3, 3, 100)
#     y_t = func(x_t)
#     ax2.plot(x_t.detach().numpy(), y_t.detach().numpy(), label="y = 4*x^2")
#
#     y_rec = [func(torch.tensor(i)).item() for i in x_rec]
#     ax2.plot(x_rec, y_rec, '-ro')
#
#     ax2.grid()
#     # ax2.set_xlim([-3, 3])
#     # 同样限制y轴，防止轨迹飞出天际
#     # ax2.set_ylim([0, 36])
#     ax2.legend()
#
#
# def main():
#     # 创建画布
#     fig = plt.figure(figsize=(16, 8))
#     # 划分为2行6列，每个学习率实验占用2列（1列画左图，1列画右图）
#     gs = gridspec.GridSpec(2, 6)
#
#     # 第一行：lr = 0.01, 0.1, 0.125
#     ax1_1 = fig.add_subplot(gs[0, 0])
#     ax1_2 = fig.add_subplot(gs[0, 1])
#     run_experiment(0.01, 4, ax1_1, ax1_2, "学习率0.01")
#
#     ax2_1 = fig.add_subplot(gs[0, 2])
#     ax2_2 = fig.add_subplot(gs[0, 3])
#     run_experiment(0.1, 4, ax2_1, ax2_2, "学习率0.1")
#
#     ax3_1 = fig.add_subplot(gs[0, 4])
#     ax3_2 = fig.add_subplot(gs[0, 5])
#     run_experiment(0.125, 4, ax3_1, ax3_2, "学习率0.125")
#
#     # 第二行：lr = 0.2, 0.3
#     ax4_1 = fig.add_subplot(gs[1, 0])
#     ax4_2 = fig.add_subplot(gs[1, 1])
#     run_experiment(0.2, 4, ax4_1, ax4_2, "学习率0.2")
#
#     ax5_1 = fig.add_subplot(gs[1, 2])
#     ax5_2 = fig.add_subplot(gs[1, 3])
#     # 0.3 时梯度爆炸，3次迭代就飞了
#     run_experiment(0.3, 4, ax5_1, ax5_2, "学习率0.3")
#
#     # 调整子图间距
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()