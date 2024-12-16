import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.arange(-10, 10, 0.1)

def leakyRelu(x):
    return np.where(x > 0, x, x*0.01)

leakyRelu = lambda x: np.where(x > 0, x, x*0.01)

y = leakyRelu(x)

# 그래프 그리기
plt.plot(x, y, label="leakyRelu")
plt.grid()
plt.show()