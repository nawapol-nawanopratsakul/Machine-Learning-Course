import numpy as np, joblib  # นำเข้า numpy (จัดการอาร์เรย์) และ joblib (โหลดไฟล์)

def run_custom_test():
    # โหลดไฟล์สเกลเลอร์จากที่ระบบเคยเซฟไว้
    scaler = joblib.load("outputs/scaler.pkl")
    # โหลดโมเดล SVM จากที่ระบบเคยเซฟไว้
    model = joblib.load("outputs/svm_model.pkl")
    
    # จำลองป้อนข้อมูลสรีระคนใหม่ 7 ค่า (อายุ, ส่วนสูง, น้ำหนัก, เวลา, ชีพจร, อุณหภูมิ, แคลอรี)
    new_data = np.array([[30, 160.0, 55.0, 15.0, 95.0, 40.0, 80.0]])
    
    # นำข้อมูลใหม่ไปปรับสเกล (transform) ก่อน แล้วโยนเข้าโมเดลเพื่อทำนาย (predict)
    prediction = model.predict(scaler.transform(new_data))
    
    # เช็กว่าถ้าผลออกมาเป็น 1 ให้โชว์ "Male" ถ้าไม่ใช่ให้โชว์ "Female"
    print("Predicted Class:", "Male" if prediction[0] == 1 else "Female")

# สั่งให้รันฟังก์ชันทันทีเมื่อกดเปิดไฟล์นี้
if __name__ == "__main__":
    run_custom_test()