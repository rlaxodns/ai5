import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
np_path = "C:\\ai5\\_data\\_save_npy\\"

x_train = np.load(np_path + "ke4507_gender_x_train.npy")
y_train = np.load(np_path + "ke4507_gender_y_train.npy")

x_train_woman = x_train[np.where(y_train > 0.0)]        # 0.0 보다 큰 y값이 있는 인덱스 추출
y_train_woman = y_train[np.where(y_train > 0.0)]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)
end1 = time.time()

## 데이터 증폭
train_datagen = ImageDataGenerator(
    # rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)
print(len(x_train_woman)) 

aug_size = 10000

randidx = np.random.randint(x_train_woman.shape[0], size = aug_size)
print(np.min(randidx), np.max(randidx))

x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()

print(x_augmented.shape)   
print(y_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=aug_size,
    shuffle = False,
).next()[0]

print(x_augmented.shape)
print(x_train.shape)

x_train = np.concatenate((x_train, x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)

print(np.unique(y_train, return_counts=True))

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), input_shape = (100,100,3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(16, (2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(254, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='acc',
    mode = 'max', 
    patience=20, 
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='acc',
    mode = 'auto', 
    verbose=1, 
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras49\\keras49_6_men_women.hdf5"
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=126,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
y_test = np.round(y_test)

acc = accuracy_score(y_test, y_pre)

print(loss[0], acc)