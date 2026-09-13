import numpy as np
import os
import joblib

def run_custom_test():
    """
    โหลดโมเดลที่เทรนเสร็จแล้วจากโฟลเดอร์ outputs มาใช้ทำนายข้อมูลใหม่
    โดยไม่ต้องเทรนซ้ำ!
    """
    print("=== Running Neural Network Prediction Test ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    model_path = os.path.join(output_dir, "nn_model.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Error: Model or Scaler not found! Please run 'main.py' first to generate them.")
        return

    # โหลด Scaler และ Model แบบสำเร็จรูป
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print("Model and Scaler loaded successfully.")
    
    # สมมติข้อมูลผู้ใช้งานใหม่ 1 คน 
    # ลำดับ: Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories
    new_data = np.array([[30, 160.0, 55.0, 15.0, 95.0, 40.0, 80.0]])
    
    # สเกลข้อมูลใหม่
    new_data_scaled = scaler.transform(new_data)
    
    # ทำนายผล
    prediction = model.predict(new_data_scaled)
    
    result = "Male" if prediction[0] == 1 else "Female"
    print("\n--- Prediction Result ---")
    print(f"Input Features : {new_data[0]}")
    print(f"Predicted Class: {result}")

if __name__ == "__main__":
    run_custom_test()