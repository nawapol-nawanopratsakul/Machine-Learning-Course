import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from cnn_model import build_cnn_model
from data_load import load_data
from preprocess import preprocess_data
from split_data import split_dataset

if __name__ == "__main__":
    print("=== กำลังประเมินผลโมเดลจากโฟลเดอร์ output ===")

    # 1. โหลดข้อมูลและแปลงข้อมูลทดสอบ
    df = load_data()
    X_scaled, y, _, _ = preprocess_data(df)
    _, X_test, _, y_test = split_dataset(X_scaled, y)

    # 2. โหลดโมเดลที่อยู่ในโฟลเดอร์ output
    output_dir = "output"
    model_path = os.path.join(output_dir, "car_cnn_model.keras")

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่ '{model_path}' กรุณารัน python main.py ก่อน!")
        exit()

    model = load_model(model_path)

    # 3. ทำนายผล
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. สร้าง DataFrame สำหรับเก็บข้อมูลผลลัพธ์
    output_df = pd.DataFrame(
        {
            "Actual_Category": y_test,
            "Predicted_Category": y_pred,
            "Confidence_Economy": y_pred_probs[:, 0],
            "Confidence_MidRange": y_pred_probs[:, 1],
            "Confidence_Luxury": y_pred_probs[:, 2],
        }
    )

    # 5. บันทึกไฟล์ CSV ลงในโฟลเดอร์ output
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "car_predictions_output.csv")
    output_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f" บันทึกไฟล์ผลลัพธ์สำเร็จ: '{output_filename}' เรียบร้อยแล้ว!")

    # 6. แสดงผลการประเมิน
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))