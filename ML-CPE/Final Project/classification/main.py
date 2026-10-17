import os
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from cnn_model import build_cnn_model
from data_load import load_data
from preprocess import preprocess_data
from split_data import split_dataset

if __name__ == "__main__":
    print(
        "=== เริ่มเทรนโมเดล 1D CNN (Regression - ทำนายราคาเป็นบาท) ==="
    )

    # 1. โหลดข้อมูล
    df = load_data()

    # 2. พรีเซ็ตข้อมูล
    X_scaled, y, scaler, label_encoders = preprocess_data(df)

    # 3. แบ่งชุดข้อมูล
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y)

    # 4. สร้างโมเดล
    input_shape = (X_train.shape[1], 1)
    model = build_cnn_model(input_shape)

    # 5. ระบบ Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("กำลังเทรนโมเดล...")
    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    # 6. สร้างโฟลเดอร์ output อัตโนมัติ
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 7. บันทึกไฟล์อาร์ทิแฟกต์ทั้งหมดลงใน output/
    model.save(os.path.join(output_dir, "car_cnn_model.keras"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(output_dir, "label_encoders.pkl"))

    print(
        f"\nเทรนและบันทึกไฟล์ทั้งหมดลงในโฟลเดอร์ '{output_dir}/' สำเร็จเรียบร้อย!"
    )