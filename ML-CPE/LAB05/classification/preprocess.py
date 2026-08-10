from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col="Sex"):

    print("Preprocessing data...")

    df[target_col] = df[target_col].map({'male': 1, 'female': 0})
    
    # แยก Features (X) และ Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data preprocessed and standardized successfully.")
    return X_scaled, y, scaler