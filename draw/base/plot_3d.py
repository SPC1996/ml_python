from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 500)
z = np.linspace(0, 2, 500)
r = z
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='curve')
ax.legend()

plt.show()
