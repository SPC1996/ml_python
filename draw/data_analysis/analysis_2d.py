import pandas
import numpy as np
import matplotlib.pyplot as plt


def load_data(x_index, y_index, data_file_name):
    data = pandas.read_csv(data_file_name)
    return data.iloc[:, x_index - 1].values, data.iloc[:, y_index - 1].values


def draw_one(xx, yy, dd, ax):
    font = {'family': 'serif',
            'color': 'red',
            'weight': 'normal',
            'size': 8,
            }
    cmap = {
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'yellow',
        5: 'pink',
        6: 'gray',
        7: 'gold',
        8: 'black',
        9: 'olive',
        10: 'purple',
    }
    ax.set_title(dd, fontdict=font)
    ax.set_xlabel('x', fontdict=font)
    ax.set_ylabel('y', fontdict=font)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
    for i in np.arange(1, 11):
        ax.scatter(xx[(i - 1) * 200 + 1:i * 200 + 1], yy[(i - 1) * 200 + 1:i * 200 + 1], c=cmap[i], s=3)
    ax.legend(['a', 'b', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g'])
    # ax.legend()


def show(x_index, y_index):
    x, y = load_data(x_index, y_index, 'train_data.csv')
    ax = plt.figure().add_subplot(111)
    draw_one(x, y, 'x:' + str(x_index) + '--y:' + str(y_index), ax)
    plt.show()


# fig = plt.figure()
# for j in np.arange(1, 13):
#     x, y = load_data(1, j, 'train_data.csv')
#     ax = fig.add_subplot(4, 3, j)
#     draw_one(x, y, 'x:' + str(1) + '--y:' + str(j), ax)


show(1, 3)
