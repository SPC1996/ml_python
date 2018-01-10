import pandas
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_file_name):
    data = pandas.read_csv(data_file_name)
    x = data[data.columns.difference(['res', 'level'])].values
    y = data['level'].values
    labels = data.columns[0:12]
    return x, y, labels


data, level, labels = load_data('train_data.csv')
angles = np.linspace(0, 2 * np.pi, data.shape[1], endpoint=False)
data = np.concatenate((data[0], [data[0][0]]))
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, data, 'bo-', linewidth=2)
ax.fill(angles, data, facecolor='r', alpha=0.25)
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
ax.set_title("Radar", va='bottom', fontproperties="SimHei")
ax.set_rlim(0, 1)
ax.grid(True)
plt.show()
print(labels)
