import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

# 데이터 생성
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50], 
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T
aaa = pd.DataFrame(aaa, columns=["Column1", "Column2"])
print("Shape of data:", aaa.shape)

# EllipticEnvelope를 사용하여 각 컬럼별 이상치 탐지
for col in aaa.columns:
    print(f"\n=== Processing {col} ===")
    
    # 데이터를 2D 배열로 변환
    col_data = aaa[[col]].values  # 데이터프레임에서 특정 컬럼을 2D로 추출
    
    # EllipticEnvelope 모델 생성
    outlier_detector = EllipticEnvelope(contamination=0.1)  # contamination: 이상치 비율
    
    # 모델 학습
    outlier_detector.fit(col_data)
    
    # 이상치 예측
    result = outlier_detector.predict(col_data)
    
    # 결과 출력
    print("Prediction:", result)
    print("Outliers:", aaa[col][result == -1].values)
