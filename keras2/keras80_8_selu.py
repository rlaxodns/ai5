import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.arange(-10, 10, 0.1)

# def selu(x, alpha=1.67326, scale=1.0507):
#     return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

selu = lambda x, alpha=1.67326, scale=1.0507: np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

y = selu(x)

# 그래프 그리기
plt.plot(x, y, label="SELU")
plt.grid()
plt.show()