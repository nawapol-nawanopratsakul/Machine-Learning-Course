import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col="Sex"):
    # 1. แปลง Label เพศเป็น 0/1 (ชาย=1, หญิง=0)
    y = df[target_col].map({'male': 1, 'female': 0}).values
    
    # 2. ปรับสเกลข้อมูล 7 ฟีเจอร์
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=[target_col]))
    
    # 3. เตรียมข้อมูล Input Layer สำหรับ CNN
    # Reshape ข้อมูลจาก 2D เป็น 3D (จำนวนแถว, 7 ฟีเจอร์, 1 แชนเนล)
    # เพื่อให้ Kernel/Filter สามารถเลื่อน (Slide) ผ่านข้อมูลเพื่อสกัด Feature ได้
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return X_cnn, y, scaler