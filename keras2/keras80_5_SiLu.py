# swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def swish(x):
#     return x*(1 / (1 + np.exp(-x)))

# x * sigmoid
# 문제점: ReLU보다 연산량이 많아서 모델이 커질수록 부담

swish = lambda x: x*(1 / (1 + np.exp(-x)))

y = swish(x)

plt.plot(x, y)
plt.grid()
plt.show()