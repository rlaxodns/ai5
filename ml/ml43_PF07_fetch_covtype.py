import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd


#1. 데이터
x, y = fetch_covtype(return_X_y=True)
random_state = 777
y = pd.get_dummies(y)

print(x.shape)
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777, stratify=y)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test  = mms.transform(x_test)

model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
y_pred = model.predict(x_test)
print("model.Score", model.score(x_test, y_test))