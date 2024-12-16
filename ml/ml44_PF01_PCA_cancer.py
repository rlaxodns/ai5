import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # 로지스틱 리그레션은 분류
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#1. 데이터 
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)
pf = PolynomialFeatures(degree=2, include_bias=False)
x = pf.fit_transform(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 1199, 
                                                    stratify = y)

print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape)   # (114, 30) (114,)

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
    model = xgb.XGBClassifier()

    #3. 훈련
    model.fit(x_train1, y_train)

    #4. 평가 예측
    y_pred = model.predict(x_test1)
    print("model.Score", model.score(x_test1, y_test))
    """
    model.Score 0.991228070175438
    """