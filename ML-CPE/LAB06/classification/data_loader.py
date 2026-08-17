import pandas as pd
import os

def load_dataset(file_path="train.csv", sample_size=5000, random_state=42):
    """
    โหลดข้อมูลจากไฟล์และสุ่มจำนวนแถว
    """
    print("Loading dataset...")
    # หาโฟลเดอร์ปัจจุบันที่ไฟล์นี้ตั้งอยู่ เพื่อแก้ปัญหา FileNotFoundError
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, file_path)
    
    df = pd.read_csv(full_path)
    
    # สุ่มข้อมูล
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    
    # ลบคอลัมน์ id ทิ้ง
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df