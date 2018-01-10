import random
import matplotlib.pyplot as plt
from numpy import *


def make_linear_separable_data(weights, num_lines):
    w = array(weights)
    num_features = len(weights)
    data_set = zeros((num_lines, num_features + 1))
    for i in range(num_lines):
        x = random.rand(1, num_features) * 20 - 10
        inner_product = sum(w * x)
        if inner_product <= 0:
            data_set[i] = append(x, -1)
        else:
            data_set[i] = append(x, 1)
    return data_set


def plot_data(data_set):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("linear separable data set")
    plt.xlabel('X')
    plt.ylabel('Y')
    idx_1 = where(data_set[:, 2] == 1)
    ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
    idx_2 = where(data_set[:, 2] == -1)
    ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=-1, s=20)
    plt.legend(loc='upper right')
    plt.show()


def train(data_set, plot=False):
    num_lines = data_set.shape[0]
    num_features = data_set.shape[1]
    w = zeros((1, num_features - 1))
    b = 0
    separated = False
    i = 0
    while not separated and i < num_lines:
        if data_set[i][-1] * sum(w * data_set[i, 0:-1]) <= 0:
            w = w + data_set[i][-1] * data_set[i, 0:-1]
            b = b + data_set[i][-1]
            separated = False
            i = 0
        else:
            i += 1
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("linear separable data set")
        plt.xlabel('X')
        plt.ylabel('Y')

        idx_1 = where(data_set[:, 2] == 1)
        ax.scatter(data_set[idx_1, 0], data_set[idx_1, 1], marker='o', color='g', label=1, s=20)
        idx_2 = where(data_set[:, 2] == -1)
        ax.scatter(data_set[idx_2, 0], data_set[idx_2, 1], marker='x', color='r', label=-1, s=20)

        w1 = w[0][0] / abs(w[0][0]) * 10
        w2 = w[0][1] / abs(w[0][0]) * 10
        b1 = b / abs(w[0][0])
        ax.annotate("vector", xy=(0, -b1 / w2), xytext=(w1 / 2, w2 / 2), size=20, arrowprops=dict(arrowstyle="-|>"))
        xx = linspace(-10, 10, 2)
        ax.plot(xx, -w1 * xx / w2 - b1 / w2, color='blue')

        plt.legend(loc='upper right')
        plt.show()

    return w, b


train_data = make_linear_separable_data([2, 5], 100)
print(make_linear_separable_data([3, 5], 100))
w, b = train(train_data, True)
print(w)
print(b)
