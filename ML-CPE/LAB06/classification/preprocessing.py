from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col="Sex"):
    """
    ทำความสะอาดข้อมูล แปลงคลาส และทำ Standardization
    """
    print("Preprocessing data...")
    # แปลงเพศเป็นตัวเลข: male = 1, female = 0
    df[target_col] = df[target_col].map({'male': 1, 'female': 0})
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # สเกลข้อมูลสำหรับ Neural Network (จำเป็นมาก)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data preprocessed and standardized successfully.")
    return X_scaled, y, scaler