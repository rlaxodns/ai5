# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# https://dacon.io/competitions/official/236068/overview/description


# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )


# 2. 모델 구성
xgbc = xgb.XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

# model = xgb.XGBClassifier()
model = VotingClassifier(
    estimators=[('XGB', xgbc), ('RF', rf), ('CAT', cat)],
    voting='hard', # 기본값
    # voting='soft' 
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print('최종점수', result)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print('acc', acc)
