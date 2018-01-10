import matplotlib.pyplot as plt
import numpy as np


def draw(plt):
    plt.axis([-12, 12, -1, 1])
    plt.grid(True)
    xx = np.linspace(-12, 12, 1000)
    plt.plot(xx, np.sin(xx), 'g-', label="$sin(x)$")
    plt.plot(xx, np.cos(xx), 'r--', label="$cos(x)$")
    plt.legend()


plt.figure()
plt1 = plt.subplot(2, 2, 1)
draw(plt1)
plt2 = plt.subplot(2, 2, 2)
draw(plt2)
plt3 = plt.subplot(2, 2, 3)
draw(plt3)
plt3 = plt.subplot(2, 2, 4)
draw(plt3)

plt.show()
