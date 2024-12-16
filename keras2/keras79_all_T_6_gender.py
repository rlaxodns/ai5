import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
np_path = "C:\\ai5\\_data\\_save_npy\\"
x_train = np.load(np_path + "ke4507_gender_x_train.npy")
y_train = np.load(np_path + "ke4507_gender_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size= 0.2, random_state=4343
)

print(x_train.shape)


from tensorflow.keras.applications import VGG19, ResNet50, ResNet101, Xception
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, NASNetMobile
from tensorflow.keras.applications import DenseNet121, MobileNetV2, EfficientNetB0

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import cv2
from sklearn.metrics import accuracy_score

# 사전 학습된 모델 리스트와 입력 크기 정의
model_list = [
    (Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3)), (71, 71)),
    (VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3)), (75, 75)),
    (InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3)), (75, 75)),
    (DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), (224, 224)),
    (EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
]

# 모델 리스트 반복 실행
for base_model, input_size in model_list:
    # 데이터 리사이즈
    x_train_resized = np.array([cv2.resize(x_img, input_size) for x_img in x_train])
    x_test_resized = np.array([cv2.resize(x_img, input_size) for x_img in x_test])

    # 새 Sequential 모델 생성
    model = Sequential()
    model.add(base_model)  # 사전 학습된 모델 추가
    model.add(GlobalAveragePooling2D())  # 글로벌 평균 풀링 레이어 추가
    model.add(Dense(100, activation='softmax'))  # CIFAR-10에 맞는 출력층 추가

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 학습
    model.fit(x_train_resized, y_train, epochs=1, batch_size=1, validation_split=0.2, verbose=0)

    # 모델 평가 및 예측
    loss, acc = model.evaluate(x_test_resized, y_test, verbose=0)


    # 정확도 출력
    print(f"Model: {base_model.name}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

