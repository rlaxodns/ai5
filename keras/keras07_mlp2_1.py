#[실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
                [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)

x = x.T

print(x.shape)
print(y.shape)

# 2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 1)

#4. 예측 및 출력
loss = model.evaluate(x, y)
result = model.predict([[13, 1.2, -1]])

print("로스는 ", loss)
print("예측값은", result)