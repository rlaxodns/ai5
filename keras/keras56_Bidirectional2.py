#52_2 복사

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape, y.shape)
# (13, 3) (13,)


x_predict = np.array([50, 60, 70])

x = x.reshape(13,3,1)
x_predict = np.array([50, 60, 70]).reshape(1,3,1)



#2. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape = (3, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    patience=50, 
    restore_best_weights=True
)

model.fit(x, y,
        epochs =300,
        batch_size=2,
        # callbacks=[es]
        )


model.save("C:\\ai5\\_save\\keras52\\keras52_LSTM2_scale03.hdf5")

#4. 평가 및 예측
loss = model.evaluate(x, y)
result = model.predict(x_predict)

print(loss, result)
