import os
import data_loader, preprocessing, split_data, cnn_model, evaluate

def main():
    print("="*60)
    print("LAB 07: การเปรียบเทียบโครงสร้าง CNN และ Epochs")
    print("="*60)
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. เตรียมข้อมูล
    df = data_loader.load_dataset("train.csv") # ใช้ไฟล์เดิมของคุณ
    X, y, scaler = preprocessing.preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    input_shape = (X_train.shape[1], 1)
    
    # ====================================================
    # การทดลองที่ 1: CNN โมเดลที่ 1 (ตื้น) + 20 Epochs
    # ====================================================
    print("\n[การทดลองที่ 1] โครงสร้างที่ 1 (1 Conv Layer) | 20 Epochs")
    model_1 = cnn_model.build_model_config_1(input_shape)
    model_1, history_1 = cnn_model.train_model(model_1, X_train, y_train, epochs=20)
    evaluate.plot_training_history(history_1, "Model_1_20Epochs")
    acc_1 = evaluate.evaluate_model(model_1, X_test, y_test)
    
    # ====================================================
    # การทดลองที่ 2: CNN โมเดลที่ 2 (ลึก) + 50 Epochs
    # ====================================================
    print("\n[การทดลองที่ 2] โครงสร้างที่ 2 (2 Conv Layers) | 50 Epochs")
    model_2 = cnn_model.build_model_config_2(input_shape)
    model_2, history_2 = cnn_model.train_model(model_2, X_train, y_train, epochs=50)
    evaluate.plot_training_history(history_2, "Model_2_50Epochs")
    acc_2 = evaluate.evaluate_model(model_2, X_test, y_test)
    
    # ====================================================
    # สรุปผลการเปรียบเทียบ (Output requirements)
    # ====================================================
    print("\n" + "="*40)
    print("สรุปเปรียบเทียบความแม่นยำ (Accuracy Comparison)")
    print("="*40)
    print(f"โมเดลที่ 1 (20 Epochs, 1 Conv Layer): {acc_1 * 100:.2f}%")
    print(f"โมเดลที่ 2 (50 Epochs, 2 Conv Layers): {acc_2 * 100:.2f}%")
    
    # แสดงตัวอย่างการทำนาย (Predictions on selected dataset)
    print("\nตัวอย่างการทำนายข้อมูล 5 รายการจาก Test Set (ด้วยโมเดลที่ 2):")
    sample_X = X_test[:5]
    sample_y = y_test[:5]
    predictions_prob = model_2.predict(sample_X, verbose=0)
    
    for i in range(5):
        pred_class = 1 if predictions_prob[i][0] > 0.5 else 0
        actual = "Male" if sample_y[i] == 1 else "Female"
        predicted = "Male" if pred_class == 1 else "Female"
        print(f"รายการที่ {i+1} | ทำนาย: {predicted} | ความเป็นจริง: {actual} "
              f"| (ความมั่นใจ {predictions_prob[i][0]:.4f})")

if __name__ == "__main__":
    main()