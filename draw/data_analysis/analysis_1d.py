import pandas
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_file_name):
    data = pandas.read_csv(data_file_name)
    x = {
        1: data['f_01'].values,
        2: data['f_02'].values,
        3: data['f_03'].values,
        4: data['f_04'].values,
        5: data['f_05'].values,
        6: data['f_06'].values,
        7: data['f_07'].values,
        8: data['f_08'].values,
        9: data['f_09'].values,
        10: data['f_10'].values,
        11: data['f_11'].values,
        12: data['f_12'].values

    }
    y = data['level'].values
    return x, y


def draw_one(xx, yy, dd, color, ax):
    font = {'family': 'serif',
            'color': 'red',
            'weight': 'normal',
            'size': 8,
            }
    ax.set_title(dd, fontdict=font)
    ax.set_xlabel('feature size', fontdict=font)
    ax.set_ylabel('chord', fontdict=font)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1,0'))
    ax.set_yticks(np.linspace(0, 10, 11))
    ax.set_yticklabels(('', 'a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g'))
    ax.scatter(xx, yy, c=color, s=5)
    ax.legend()


def draw_all(x, y, wsize, hsize, figure):
    for i in np.arange(1, 13):
        ax = figure.add_subplot(wsize, hsize, i)
        draw_one(x[i], y, 'feature ' + str(i), 'green', ax)


X, Y = load_data('train_data.csv')
fig = plt.figure()
fig.subplots_adjust(hspace=0.8, wspace=0.3)
draw_all(X, Y, 4, 3, fig)

plt.show()
