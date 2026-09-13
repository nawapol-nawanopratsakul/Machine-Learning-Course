import data_load, preprocess, split_data, svm_model, evaluate  # ดึงโมดูลทั้ง 5 ตัวด้านบนเข้ามา
import os, joblib  # ดึงเครื่องมือจัดการไฟล์และเซฟโมเดล

def main():
    # สร้างโฟลเดอร์ 'outputs'
    os.makedirs("outputs", exist_ok=True)
    
    # 1. โหลดข้อมูล
    df = data_load.load_dataset()
    # 2. ปรับสเกลข้อมูล
    X, y, scaler = preprocess.preprocess_data(df)
    # 3. แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    
    # 4. สร้างและเทรนโมเดล SVM
    model = svm_model.train_svm(X_train, y_train)
    # 5. ประเมินผลและสร้างกราฟ
    evaluate.evaluate_model(model, X_test, y_test)
    
    # 6. เซฟเครื่องมือปรับสเกล และ ตัวโมเดล เป็นไฟล์ .pkl เพื่อนำไปใช้งานต่อบน Production
    joblib.dump(scaler, "outputs/scaler.pkl")
    joblib.dump(model, "outputs/svm_model.pkl")
    
    print("Pipeline Complete! Files saved in 'outputs/'")

# เช็กว่าไฟล์นี้ถูกกดรันโดยตรงหรือไม่ ถ้าใช่ให้รันฟังก์ชัน main()
if __name__ == "__main__":
    main()