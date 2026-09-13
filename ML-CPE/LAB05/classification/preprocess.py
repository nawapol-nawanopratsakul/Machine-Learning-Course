from sklearn.preprocessing import StandardScaler  # นำเข้าเครื่องมือปรับสเกลข้อมูล

def preprocess_data(df, target_col="Sex"):
    # เปลี่ยนคำว่า 'male' เป็น 1 และ 'female' เป็น 0 แล้วเก็บลงตัวแปร y (Target)
    y = df[target_col].map({'male': 1, 'female': 0})
    
    # เรียกใช้งานเครื่องมือปรับสเกล
    scaler = StandardScaler()
    
    # ตัดคอลัมน์เพศออก แล้วส่งเข้าเครื่องปรับสเกลทันที พร้อมคืนค่า X, y, และ scaler
    return scaler.fit_transform(df.drop(columns=[target_col])), y, scaler