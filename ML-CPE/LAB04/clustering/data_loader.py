from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ถอยหลัง 1 ชั้น (.parent.parent) เพื่อเข้าไปที่โฟลเดอร์ data
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
NUMERIC_FEATURES = ["Weight", "Height", "Age", "Duration", "Heart_Rate", "Body_Temp", "Calories"]

def load_data(n_samples=5000, seed=42):
    df = pd.read_csv(CSV_PATH).dropna()
    
    # สุ่มข้อมูลเพื่อไม่ให้กิน RAM เยอะเกินไปตอนรัน K-Means
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=seed)

    X_raw = df[NUMERIC_FEATURES].to_numpy(dtype="float32")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype("float32")

    return {
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "feature_names": NUMERIC_FEATURES,
        "dataframe": df
    }

if __name__ == "__main__":
    data = load_data()
    print("X_raw shape    :", data["X_raw"].shape)