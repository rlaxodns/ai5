import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

np.random.seed(7777)
x = 2*np.random.rand(100, 1) -1 # -1 ~ 1 까지의 난수 생성
print(x)
print(np.max(x), np.min(x))

y = 3*x**2 + 2*x + 1 + np.random.randn(100, 1) # y = 3x^2 + 2x + 1 + 노이즈

# rand : 0~1 사이의 균등분포
# randn : 평균 0, 표준편차 1 정규분포의 난수

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)
print(x_poly.shape)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)

# x-y 관계 
plt.scatter(x, y, color = 'blue', label = 'original_data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("poly regression example")

# plt.show()

# 다항식 회귀 그래프
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_poly = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot2 = model2.predict(x_test_poly)

plt.plot(x_test, y_plot, color = 'red', label = '기냥')
plt.plot(x_test, y_plot2, color = 'blue', label = 'poly reg')

plt.legend()
plt.show()

# plt.scatter(x_poly, y, color = 'red')