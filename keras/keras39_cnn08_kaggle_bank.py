# DNN -> CNN
# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\bank\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isna().sum())

# 문자열 데이터 수치화
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(train_csv)

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = train_csv['Exited']


print(x.shape)  # (165034, 10)
print(y.shape)  # (165034,)

x = x.to_numpy()
x = x.reshape(165034, 5, 2, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5324)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(5,2,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu', strides=1, padding='same'))    
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es],
          )
end = time.time()
model.save("C:\\ai5\\_save\\keras39\\keras39_08_bank.hdf5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

# print('acc :', round(loss[1],3))    # metrix 에서 설정한 값 반환   

y_pred = model.predict(x_test)

# r2 = r2_score(y_test, y_pred)
# print('r2 score :', r2)
print(y_pred)
y_pred = np.round(y_pred) 
accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
# print("걸린 시간 :", round(end-start,2),'초')
print("걸린 시간 :", round(end-start,2),'초')