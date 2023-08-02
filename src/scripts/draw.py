import matplotlib.pyplot as plt
import numpy as np


def plot_2d_list(data, predict_period, title, fig_width=8, fig_height=6):
    num_users = len(data)
    num_iterations = len(data[0])

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 绘制每个用户的数据曲线
    for user in range(num_users):
        user_data = data[user]
        x = np.arange(0, num_iterations * predict_period, predict_period)
        y = user_data
        y = [i for i in y]
        ax.plot(x, y, label='V{}'.format(user + 1))

    # 添加图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('Communication rounds')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=0.5)

    plt.title(title)
    plt.savefig(title + '.svg', format="svg")
