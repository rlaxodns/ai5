import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
# print(x_data.shape, y_data.shape) #(4, 2) (4,)

#2. 모델
model = SVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가 및 예측
acc= model.score(x_data, y_data)
print(acc)

y_pre = model.predict(x_data)
acc2 = accuracy_score(y_data, y_pre)
print(acc2)

print('========================================')
print(y_data)
print(y_pre)

# SVC의 다층구성을 통해서 가볍게 해결
# 1.0
# 1.0
# ========================================
# [0 1 1 0]
# [0 1 1 0]