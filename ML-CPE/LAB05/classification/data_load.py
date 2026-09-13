import pandas as pd, os  # นำเข้าไลบรารี pandas (จัดการตาราง) และ os (จัดการไฟล์)

def load_dataset(file_path="train.csv", sample_size=5000):
    # ค้นหาที่อยู่ไฟล์แบบอัตโนมัติ ป้องกันปัญหาหาไฟล์ไม่เจอเมื่อย้ายโฟลเดอร์
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    # อ่านไฟล์ CSV เข้ามาเป็นตาราง (DataFrame)
    df = pd.read_csv(path)
    
    # ถ้าข้อมูลมีเกิน 5,000 แถว ให้สุ่มมาแค่นั้น ถ้าไม่ถึงก็ใช้ทั้งหมด
    df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    
    # ลบคอลัมน์ 'id' ทิ้ง (ถ้ามี) แล้วส่งคืนตารางข้อมูลกลับไปให้ระบบ
    return df.drop(columns=['id'], errors='ignore')