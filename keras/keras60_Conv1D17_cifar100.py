from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.3,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 40000   # 증가시킬 사이즈 

randidx = np.random.randint(x_train.shape[0], size = augment_size) # 60000, size = 40000
print(randidx)
print(np.min(randidx), np.max(randidx)) 

print(x_train[0].shape) #(28, 28)

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)  # (40000, 28, 28, 1)

x_train = x_train.reshape(
    x_train.shape[0], 
    x_train.shape[1], 
    x_train.shape[2], 
    x_train.shape[3]
)
x_test = x_test.reshape( x_test.shape[0], 
    x_test.shape[1], 
    x_test.shape[2], 
    x_test.shape[3])

print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)

##인코딩
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    x_train.shape[2]*x_train.shape[3]
)

x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    x_test.shape[2]*x_test.shape[3]
)

print(x_train.shape, x_test.shape) #(50000, 32, 96) (10000, 32, 96)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, LSTM, Conv1D
model = Sequential()
model.add(Conv1D(32, 3, input_shape = (32,96), activation='relu'))

model.add(Dropout(0.2))


model.add(Conv1D(32, 3, ))
model.add(Dropout(0.2))

model.add(Conv1D(16, 2))
model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(254, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32))
model.add(Dense(100, activation='softmax'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', 
    patience=20, 
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto', 
    verbose=1, 
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras49\\keras49_1_fashion.hdf5"
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=512,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.argmax(model.predict(x_test), axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_pre)

print(loss[0], acc)
