import os
import joblib
import data_loader, preprocessing, split_data, cnn_model, evaluate

def main():
    print("="*60)
    print("🚀 เริ่มต้นระบบจำแนกเพศด้วย Deep Learning (1D CNN)")
    print("="*60)
    
    os.makedirs("outputs", exist_ok=True)
    
    df = data_loader.load_dataset("train.csv")
    X, y, scaler = preprocessing.preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    
    model = cnn_model.train_cnn(X_train, y_train, epochs=20)
    evaluate.evaluate_model(model, X_test, y_test)
    
    joblib.dump(scaler, "outputs/scaler.pkl")
    model.save("outputs/cnn_model.keras") 
    
    print("\n✅ Pipeline Complete! บันทึกไฟล์พร้อมใช้ไว้ที่ 'outputs/'")

if __name__ == "__main__":
    main()