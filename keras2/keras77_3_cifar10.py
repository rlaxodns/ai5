import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

#x데이터 스케일링
x_train = x_train/255.
x_test = x_test/255.

# y원핫인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print(y_train.shape) #(50000, 10)

# #2. 모델 구성
model = Sequential()
model.add(Conv2D(50, (4,4),
                  input_shape = (32,32,3),
                  padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(25, (3,3)))
model.add(BatchNormalization())
# batch normalization은 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도
# 각 배치별로 평균과 분산을 이용해 정규화하는 것을 뜻
# 정규화는 데이터의 중복을 허용하지 않도록 하는 것
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min', 
    patience=50, 
    verbose=1,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto', 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras35//keras35__save_cnn6_mnist00.hdf5"
)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
          epochs = 10000,
          batch_size=128,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)

print("loss :", loss,
      "acc :", acc)

# loss : [1.199658751487732, 0.5625] acc : 0.5625