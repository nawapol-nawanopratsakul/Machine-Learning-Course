import os
import re
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df):
  # 1. เพิ่มฟังก์ชันทำความสะอาดคอลัมน์ year ให้เป็นตัวเลข (ป้องกันข้อมูลที่เป็นข้อความปะปน)
  def parse_year(val):
    try:
      return int(val)
    except (ValueError, TypeError):
      # ถ้าแปลงตรงๆ ไม่ได้ ให้ลองค้นหาตัวเลขปี 4 หลักจากข้อความ
      match = re.search(r"\b(19\d{2}|20\d{2})\b", str(val))
      if match:
        return int(match.group(1))
      return 2020  # ค่าสำรองกรณีแปลงไม่ได้

  if "year" in df.columns:
    df["year"] = df["year"].apply(parse_year)

  # 2. ถ้ายังไม่มีคอลัมน์ brand แต่มีคอลัมน์ name ให้ตัดคำแรกมาเป็น brand อัตโนมัติ
  if "brand" not in df.columns and "name" in df.columns:
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0])

  # 3. ถ้ายังไม่มีคอลัมน์ car_age แต่มี year ให้คำนวณอายุรถ (ปีปัจจุบัน 2026)
  if "car_age" not in df.columns and "year" in df.columns:
    current_year = 2026
    df["car_age"] = current_year - df["year"]

  # กำหนดรายชื่อฟีเจอร์ที่ต้องการใช้งานจริง
  features = [
      "brand",
      "year",
      "car_age",
      "km_driven",
      "fuel",
      "seller_type",
      "transmission",
      "owner",
  ]

  # เลือกเฉพาะฟีเจอร์ที่มีอยู่จริงใน DataFrame เพื่อป้องกัน Error
  available_features = [f for f in features if f in df.columns]
  df_filtered = df[available_features].copy()

  # แปลงคอลัมน์ที่เป็นข้อความด้วย LabelEncoder
  label_encoders = {}
  cat_cols = df_filtered.select_dtypes(include=["object"]).columns

  for col in cat_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
    label_encoders[col] = le

  X = df_filtered
  y = df["selling_price"].values

  # สเกลเฉพาะข้อมูลขาเข้า (X)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  return X_scaled, y, scaler, label_encoders