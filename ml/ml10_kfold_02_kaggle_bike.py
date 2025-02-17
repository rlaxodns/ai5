# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
# 현재는 결측치만 확인하고 있으나 추후 이상치 또한 확인해야함
# 

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVR

#1. 데이터
path = "C:/Users/kim/Downloads/bike-sharing-demand/"  

train_csv = pd.read_csv(path+"train.csv", index_col=0)  # 첫 번쨰 데이터는 시간 데이터이기 때문에 분할의 필요하여 복잡하니 우선은 인덱스로 잡기
test_csv = pd.read_csv(path+"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv", index_col=0)

#########결측치 확인############
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

## x와 y를 분리
x = train_csv.drop(['casual', 'registered','count'], axis=1) 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=777)

kfold = KFold(
    n_splits=5, 
    shuffle=True,
    random_state=777
)

#2. 모델
model = SVR()

#3. 컴파일 및 훈련
score = cross_val_score(model, x_train, y_train,
                        cv= kfold)
print("score", score, "평균score", np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)
r2 = r2_score(y_test, y_pre)
print(r2)
# score [0.18221553 0.19149334 0.2080923  0.20811642 0.20211456] 평균score 0.19840643047005518
# score [0.18258067 0.17977328 0.19469273 0.19488044 0.20529901] 평균score 0.19144522770880995
# 0.09901723112964078