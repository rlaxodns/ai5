from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
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
print(x.shape)
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777,)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

#pca
from sklearn.decomposition import PCA
pca = PCA()
x_train1 = pca.fit_transform(x_train)
x_test1 = pca.transform(x_test)

cum = np.cumsum(pca.explained_variance_ratio_)

num = [np.argmax(cum)+1,np.argmax(cum>= 0.999)+1,
       np.argmax(cum>= 0.99)+1,np.argmax(cum>= 0.95)+1 ]
for i in range(0, len(num), 1):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    #2. 모델
    model = XGBRegressor()

    #3. 훈련
    model.fit(x_train1, y_train)

    #4. 평가 예측
    y_pred = model.predict(x_test1)
    print("model.Score", model.score(x_test1, y_test))