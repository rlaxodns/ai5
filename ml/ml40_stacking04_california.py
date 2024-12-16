from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingRegressor, VotingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor


#1. 데이터
x, y = fetch_california_housing(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777,)

#2. 모델
xgbc = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

model = StackingRegressor(
    estimators=[("XGB", xgbc), ("RF", rf), ("CAT", cat)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv = 5
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
y_pred = model.predict(x_test)
print("model.Score", model.score(x_test, y_test))
print("스태킹 r2", r2_score(y_test, y_pred))