import data_loader, preprocessing, split_data as split_data, nn_model, evaluate as evaluate
import os, numpy as np, joblib, json

def main():
    print("=== Start Neural Network Pipeline ===")
    os.makedirs("outputs", exist_ok=True) # สร้างโฟลเดอร์สำหรับเก็บไฟล์ทั้งหมด

    # เรียกใช้ฟังก์ชันโหลดข้อมูล, พรีโปรเซส, และแบ่ง Train/Test ตามลำดับ
    df = data_loader.load_dataset("train.csv", sample_size=5000)
    X, y, scaler = preprocessing.preprocess_data(df, target_col="Sex")
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    
    print("Saving artifacts to 'outputs/'...")
    # นำตัวแปรชุดข้อมูลทั้งหมดมาจัดกลุ่มใน Dictionary เพื่อง่ายต่อการเซฟ
    arrays = {"features": X, "labels": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    # ใช้ลูป for ทยอยเซฟข้อมูลทีละตัวเป็นไฟล์ .npy 
    for name, data in arrays.items():
        np.save(f"outputs/{name}.npy", data)
        
    # เซฟข้อมูลพจนานุกรมคลาสเป็นไฟล์ .json
    json.dump({"0": "female", "1": "male"}, open("outputs/classes.json", "w"))
    # เซฟตัวปรับสเกล (Scaler) เป็นไฟล์ .pkl สำหรับใช้บน Production
    joblib.dump(scaler, "outputs/scaler.pkl")

    # สั่งเทรนโมเดล และ ประเมินผล
    model = nn_model.train_nn(X_train, y_train)
    acc = evaluate.evaluate_model(model, X_test, y_test)
    # เซฟโมเดลที่เทรนเสร็จแล้วเป็นไฟล์ .pkl
    joblib.dump(model, "outputs/nn_model.pkl")
    
    print(f"\nPipeline Complete! NN Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# เช็กว่าไฟล์นี้ถูกกดรันโดยตรงหรือไม่ ถ้าใช่ให้เรียกฟังก์ชัน main() ทันที
if __name__ == "__main__":
    main()