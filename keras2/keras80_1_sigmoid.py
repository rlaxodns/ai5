import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1/ (1 + np.exp(-x))

sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)

print(x)

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()