import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.title('single variable')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0, 5, 0, 10])
plt.grid(True)
xx = np.linspace(0, 5, 10)
plt.plot(xx, 2 * xx, 'g-')
plt.show()
