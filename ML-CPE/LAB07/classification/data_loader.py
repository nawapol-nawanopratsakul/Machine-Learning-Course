import pandas as pd
import os

def load_dataset(file_path="train.csv", sample_size=5000):
    # ค้นหาที่อยู่ไฟล์อัตโนมัติ (ใช้ไฟล์ train_2.csv)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    df = pd.read_csv(path)
    
    # สุ่มข้อมูลเพื่อลดภาระเครื่อง
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    # ลบคอลัมน์ id ทิ้ง
    return df.drop(columns=['id'], errors='ignore')