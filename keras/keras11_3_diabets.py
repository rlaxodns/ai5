from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=72)

#2. 모델
model = Sequential()
model.add(Dense(1000, input_dim=10))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4. 에측 밒 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print(loss)
print(r2)

"""
random_state=72
epochs=500, batch_size=1
2212.451171875
0.6306960472982034
"""