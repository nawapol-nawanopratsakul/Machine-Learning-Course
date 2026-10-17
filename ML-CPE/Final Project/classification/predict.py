import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ใช้เส้นทางอ้างอิงจากตัวไฟล์สคริปต์
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output")

model = load_model(os.path.join(output_dir, "car_cnn_model.keras"))
scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))

def predict_batch_cars(car_list_dicts):
    # แปลงข้อมูลทั้งหมดเป็น DataFrame ทีเดียว
    df = pd.DataFrame(car_list_dicts)

    # 1. สร้างคอลัมน์ brand อัตโนมัติถ้ามี name
    if "brand" not in df.columns and "name" in df.columns:
        df["brand"] = df["name"].apply(lambda x: str(x).split()[0])

    # 2. คำนวณ car_age อัตโนมัติถ้ามี year (ปีปัจจุบัน 2026)
    if "car_age" not in df.columns and "year" in df.columns:
        current_year = 2026
        df["car_age"] = current_year - df["year"]

    # แปลงคอลัมน์ข้อความด้วย LabelEncoder (รองรับข้อมูลหลายแถว)
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda val: val if val in encoder.classes_ else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    # จัดเรียงลำดับคอลัมน์ให้ตรงกับตอนเทรนอัตโนมัติ
    if hasattr(scaler, "feature_names_in_"):
        df = df[scaler.feature_names_in_]

    # สเกลและปรับมิติข้อมูลสำหรับ 1D CNN (รองรับ Batch หลายแถว)
    X_scaled = scaler.transform(df)
    X_input = np.expand_dims(X_scaled, axis=2)

    # ทำนายราคารวดเดียวทุกคัน
    predictions = model.predict(X_input)

    # จัดรูปแบบผลลัพธ์ส่งกลับเป็นรายคัน
    results = []
    for i, pred in enumerate(predictions):
            # ปรับลดราคาลงครึ่งหนึ่งตรงนี้
            price = float(pred[0]) * 1.3

            car_name = car_list_dicts[i].get("name", f"รถคันที่ {i+1}")
            results.append(
                {"car": car_name, "estimated_price": f"{price:,.2f} บาท"}
            )

    return results

if __name__ == "__main__":
    # 🚗 ตัวอย่างข้อมูลรถ 10 คัน
    sample_10_cars = [
        {"name": "Maruti Swift Dezire VDI", "year": 2014, "km_driven": 145000, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
        {"name": "Hyundai Grand i10 Magna", "year": 2015, "km_driven": 20000, "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
        {"name": "Honda City 1.5 S MT", "year": 2018, "km_driven": 35000, "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
        {"name": "Toyota Fortuner 2.8 V", "year": 2019, "km_driven": 60000, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"},
        {"name": "Ford EcoSport Trend", "year": 2016, "km_driven": 55000, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Second Owner"},
        {"name": "Mahindra Scorpio S10", "year": 2017, "km_driven": 90000, "fuel": "Diesel", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
        {"name": "Tata Nexon XZ Plus", "year": 2020, "km_driven": 15000, "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
        {"name": "Renault Kwid RXT", "year": 2018, "km_driven": 25000, "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
        {"name": "BMW 3 Series 320d", "year": 2015, "km_driven": 70000, "fuel": "Diesel", "seller_type": "Luxury Car Dealer", "transmission": "Automatic", "owner": "Second Owner"},
        {"name": "Volkswagen Vento Highline", "year": 2016, "km_driven": 65000, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner"}
    ]

    print("🔍 กำลังประเมินราคารวม 10 คัน...")
    results = predict_batch_cars(sample_10_cars)
    
    print("\nผลการทำนายราคารถทั้ง 10 คัน:")
    for idx, res in enumerate(results, 1):
        print(f"{idx}. {res['car']} ➔ ราคาประเมิน: {res['estimated_price']}")