#49_1 복붙

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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
# 이미지를 저장한 것을 확인한 후에 이상한 것이 있는지에 대해서 확인하는 절차를 가지는 것이 좋다


##데이터 증폭##

augment_size = 40000   # 증가시킬 사이즈 

randidx = np.random.randint(x_train.shape[0], size = augment_size) # 60000, size = 40000
print(randidx)
print(np.min(randidx), np.max(randidx)) 

print(x_train[0].shape) #(28, 28)

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],    #40000
    x_augmented.shape[1],    #28
    x_augmented.shape[2], 1) #28, 1
print(x_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir="C:\\ai5\\_data\\_save_img\\01_fashion"
).next()[0]

print(x_augmented.shape)  # (40000, 28, 28, 1)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)

##인코딩
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)