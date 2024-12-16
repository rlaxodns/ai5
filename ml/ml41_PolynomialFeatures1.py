import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=True == 기본값, Linear Regressor계열의 모델의 경우에서는 bias True가 성능이 좋다
x_pf = pf.fit_transform(x)
print(x_pf)
"""
[[ 0.  1.  0.  0.  1.]
 [ 2.  3.  4.  6.  9.]
 [ 4.  5. 16. 20. 25.]
 [ 6.  7. 36. 42. 49.]]
 """
# 통상적으로 선형모델(lr등)에 쓸 경우에는 include_bias=True를 써서 1만 있는 컬럼을 만드는게 좋고,
# 왜냐하면 y = wx + b의 바이어스 = 1의 역할을 하기 때문
# 비선형모델(rf, xgb등)에 쓸 경우에는 include_bias=False가 좋다

pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf2 = pf.fit_transform(x)
print(x_pf2)
"""
[[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
 [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
 [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
 [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]
"""