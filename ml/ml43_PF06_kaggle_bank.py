# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

print(train.dtypes)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']

print(x.shape)
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
print(x.shape)

print(x.shape, y.shape) #(165034, 10) (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    random_state= 6666)

model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
y_pred = model.predict(x_test)
print("model.Score", model.score(x_test, y_test))