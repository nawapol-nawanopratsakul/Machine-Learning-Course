import os
import joblib
import data_loader, preprocessing, split_data, cnn_model, evaluate

def main():
    print("="*60)
    print("LAB 07: เทรนโมเดลเปรียบเทียบและสร้างกราฟระดับวิจัย")
    print("="*60)
    
    os.makedirs("outputs", exist_ok=True)
    
    df = data_loader.load_dataset("train.csv")
    X, y, scaler = preprocessing.preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    input_shape = (X_train.shape[1], 1)
    
    EPOCHS = 50 
    
    print(f"\n[เทรนโมเดล 1] 1 Conv Layer | {EPOCHS} Epochs")
    model_1 = cnn_model.build_model_config_1(input_shape)
    model_1, history_1 = cnn_model.train_model(model_1, X_train, y_train, epochs=EPOCHS)
    acc_1 = evaluate.evaluate_model(model_1, X_test, y_test, "Model_1")
    
    print(f"\n[เทรนโมเดล 2] 2 Conv Layers | {EPOCHS} Epochs")
    model_2 = cnn_model.build_model_config_2(input_shape)
    model_2, history_2 = cnn_model.train_model(model_2, X_train, y_train, epochs=EPOCHS)
    acc_2 = evaluate.evaluate_model(model_2, X_test, y_test, "Model_2")
    
    # วาดกราฟรวมแบบ Paper วิจัย
    evaluate.plot_paper_style_history(history_1, history_2, ["CNN Model 1 (Light)", "CNN Model 2 (Deep)"])
    
    # เซฟเครื่องมือและโมเดล (เลือกเซฟโมเดลที่ 2 เพราะโครงสร้างลึกกว่า)
    joblib.dump(scaler, "outputs/scaler.pkl")
    model_2.save("outputs/cnn_model.keras")
    
    print("\n" + "="*40)
    print("📊 สรุปความแม่นยำบนข้อมูลทดสอบ (Test Set)")
    print("="*40)
    print(f"โมเดลที่ 1: {acc_1 * 100:.2f}%")
    print(f"โมเดลที่ 2: {acc_2 * 100:.2f}%")
    print("\n✅ บันทึกไฟล์ทั้งหมด (กราฟ, โมเดล, Scaler) ไว้ที่โฟลเดอร์ 'outputs/' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()