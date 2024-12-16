# https://dacon.io/competitions/official/236068/overview/description

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,VotingClassifier,RandomForestClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)


x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777, 
                                                    stratify=y)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)


# 2. 모델 구성
xgbc = XGBClassifier()
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
