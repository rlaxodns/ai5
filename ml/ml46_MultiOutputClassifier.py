from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
np.random.seed(777)

def creat_multiclass_data_with_labels():
    x = np.random.rand(20, 3)
    y = np.random.randint(0, 5, size = (20, 3))

    x_df = pd.DataFrame(x, columns=['Features1', 'Features2', 'Features3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])

    return x_df, y_df


x, y = creat_multiclass_data_with_labels()
print(x.shape, y.shape)

print(x) # 0.152664   0.302357   0.062036
print(y) #       1       3       0
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.multioutput import MultiOutputClassifier

#2. 모델
# model = RandomForestClassifier()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어: ", 
#       round(mean_absolute_error(y, y_pred), 4)) #RandomForestClassifier 스코어:  0.0
# print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[0 8 1]]

# model = MultiOutputClassifier(LogisticRegression())
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어: ", 
#       round(mean_absolute_error(y, y_pred), 4)) #MultiOutputClassifier 스코어:  2.35
# print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[1 6 6]]

# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어: ", 
#       round(mean_absolute_error(y, y_pred), 4)) #Ridge 스코어:  2.0382
# print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[161.94258184 -22.94176826 -18.45351375]]

# model = Lasso()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어: ", 
#       round(mean_absolute_error(y, y_pred), 4)) #Lasso 스코어:  2.475
# print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[4.65 4.8  5.  ]]

# # model = XGBClassifier() #error
# # model.fit(x, y)
# # y_pred = model.predict(x)
# # print(model.__class__.__name__, "스코어: ", 
# #       round(mean_absolute_error(y, y_pred), 4)) #XGBRegressor 스코어:  0.0008
# # print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[138.0005    33.002136  67.99897 ]]

# model = MultiOutputClassifier(XGBClassifier()) 
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어: ", 
#       round(mean_absolute_error(y, y_pred), 4)) #MultiOutputClassifier 스코어:  0.0
# print(model.predict([[0.723264,   0.787623,   0.110613]])) # [[3 3 0]]

model2 = MultiOutputClassifier(CatBoostClassifier()) #error
model2.fit(x, y)
y_pred = model2.predict(x)
print(model2.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred.reshape(20,3)), 4)) #CatBoostRegressor 스코어:  0.0638
print(model2.predict([[0.723264,   0.787623,   0.110613]])) #[[[1 1 4]]]


model = MultiOutputClassifier(LGBMClassifier()) 
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어: ", 
      round(mean_absolute_error(y, y_pred), 4)) #MultiOutputClassifier 스코어:  2.6167
print(model.predict([[2, 110, 43]])) # [[2 5 3]]