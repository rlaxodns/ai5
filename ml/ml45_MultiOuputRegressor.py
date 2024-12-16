import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

#1. 데이터
x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape)

#       x                    y
# [5.  162.  60.]  -> [191.  36.  50.]
# [2.  110.  43.]  -> [138.  33.  68.]

#2. 모델
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #RandomForestRegressor 스코어:  3.7202
print(model.predict([[2, 110, 43]])) # [[152.94  34.1   64.08]]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #LinearRegression 스코어:  7.4567
print(model.predict([[2, 110, 43]])) # [[187.33745435  37.08997099  55.40216714]]

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #Ridge 스코어:  7.4569
print(model.predict([[2, 110, 43]])) # [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #XGBRegressor 스코어:  0.0008
print(model.predict([[2, 110, 43]])) # [[138.0005    33.002136  67.99897 ]]

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

model = MultiOutputRegressor(LGBMRegressor(verbose = 0))
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #MultiOutputRegressor 스코어:  8.91
print(model.predict([[2, 110, 43]])) #[[178.6  35.4  56.1]]
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

model = CatBoostRegressor(verbose=0, loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #CatBoostRegressor 스코어:  0.0638
print(model.predict([[2, 110, 43]])) #[[138.21649371  32.99740595  67.8741709 ]]
# Currently only multi-regression, multilabel and survival objectives work with multidimensional target