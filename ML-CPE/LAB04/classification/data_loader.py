from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ชี้ไปที่โฟลเดอร์ data ซึ่งอยู่ระดับเดียวกับ classification
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
TARGET = "Sex"
NUMERIC_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]

def load_data(test_size=0.2, seed=42, n_samples=5000):
    df = pd.read_csv(CSV_PATH).dropna()            
    
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=seed)

    X = df[NUMERIC_FEATURES].copy()
    class_names = sorted(df[TARGET].unique())
    y = df[TARGET].map({name: i for i, name in enumerate(class_names)})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES,
        "n_rows": len(df),
    }

if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)