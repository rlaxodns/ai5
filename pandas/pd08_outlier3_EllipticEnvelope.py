import numpy as np
import pandas as pd
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
print(aaa.shape)  # 13,
aaa = aaa.reshape(-1, 1) # 13, 1

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope()

outlier.fit(aaa)
result = outlier.predict(aaa)
print(result)