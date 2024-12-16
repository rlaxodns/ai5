import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x = np.arange(-5, 5, 0.1)

# ELU 함수 정의
# def elu(x, alpha=1.0):
#     return np.where(x > 0, x, alpha * (np.exp(x) - 1))

elu = lambda x, alpha=1.0: np.where(x>0, x, alpha*(np.exp(x)-1))

y = elu(x)

# 그래프 그리기
plt.plot(x, y, label="ELU")
plt.grid()
plt.show()