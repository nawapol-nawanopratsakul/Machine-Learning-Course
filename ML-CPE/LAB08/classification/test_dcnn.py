import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model

def run_custom_test():
    print("==================================================")
    print("Testing new user data with DCNN model")
    print("==================================================")
    
    if not os.path.exists("outputs/scaler.pkl") or not os.path.exists("outputs/dcnn_model.keras"):
        print("Error: Model files not found. Please run main.py first.")
        return

    scaler = joblib.load("outputs/scaler.pkl")
    model = load_model("outputs/dcnn_model.keras")
    print("DCNN Model and Scaler loaded successfully.\n")
    
    # [Age, Height, Weight, Duration, Heart_Rate, Body_Temp, Calories]
    new_data = np.array([[28, 175.0, 72.0, 25.0, 105.0, 39.5, 130.0]])
    
    new_data_scaled = scaler.transform(new_data)
    new_data_dcnn = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
    
    prediction_prob = model.predict(new_data_dcnn, verbose=0)
    
    predicted_class = 1 if prediction_prob[0][0] > 0.5 else 0
    predicted_label = 'Male' if predicted_class == 1 else 'Female'
    
    print("New user data input:")
    print(f"- Age: {new_data[0][0]} | Height: {new_data[0][1]} cm | Weight: {new_data[0][2]} kg")
    print("-" * 50)
    print(f"Prediction Result: {predicted_label}")
    print(f"Confidence (Probability): {prediction_prob[0][0]*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_custom_test()