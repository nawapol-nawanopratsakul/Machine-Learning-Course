import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model

def run_custom_test():
    print("=== ทดสอบจำลองการทำนายผู้ใช้ใหม่ด้วย CNN ===")
    
    scaler = joblib.load("outputs/scaler.pkl")
    model = load_model("outputs/cnn_model.keras")
    
    # ข้อมูลจำลอง (Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories)
    new_data = np.array([[30, 160.0, 55.0, 15.0, 95.0, 40.0, 80.0]])
    
    new_data_scaled = scaler.transform(new_data)
    new_data_cnn = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
    
    prediction_prob = model.predict(new_data_cnn, verbose=0)
    predicted_class = 1 if prediction_prob[0][0] > 0.5 else 0
    
    print(f"Predicted Class: {'Male' if predicted_class == 1 else 'Female'} " 
          f"(ความมั่นใจ: {prediction_prob[0][0]*100:.2f}%)")

if __name__ == "__main__":
    run_custom_test()