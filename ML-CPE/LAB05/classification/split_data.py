from sklearn.model_selection import train_test_split  # นำเข้าเครื่องมือแบ่งชุดข้อมูล

def split_dataset(X, y, test_size=0.2):
    # แบ่งข้อมูลทดสอบ 20% และใช้ stratify=y เพื่อล็อกให้สัดส่วนชาย/หญิงสมดุลกันเป๊ะๆ
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)