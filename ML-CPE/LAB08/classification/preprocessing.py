import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col="Sex"):
    y = df[target_col].map({'male': 1, 'female': 0}).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=[target_col]))
    
    X_dcnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return X_dcnn, y, scaler