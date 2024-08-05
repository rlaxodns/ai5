# DNN -> CNN

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (200000, 200)
print(y.shape)  # (200000,)

print(pd.value_counts(y, sort=True))    # 이진 분류
# 0    179902
# 1     20098

x = x.to_numpy()
x = x.reshape(200000, 25 , 8, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5233,
                                                    stratify=y)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(25,8,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
# model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es,],
          )
end = time.time()
model.save("C:\\ai5\\_save\\keras39\\keras39_12_santander.hdf5")

#4. 평가, 예측
loss = model.evaluate(x_test,y_test,verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))


y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

print("걸린 시간 :", round(end-start,2),'초')
