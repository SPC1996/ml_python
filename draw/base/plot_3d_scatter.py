from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
xx = np.linspace(0, 5, 10)
yy = np.linspace(0, 5, 10)
zz1 = xx
zz2 = 2 * xx
zz3 = 3 * xx
ax.scatter(xx, yy, zz1, c='red', marker='o')
ax.scatter(xx, yy, zz2, c='green', marker='^')
ax.scatter(xx, yy, zz3, c='black', marker='*')
ax.legend()

plt.show()
