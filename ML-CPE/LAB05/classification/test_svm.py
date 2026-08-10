import numpy as np
import data_load
import preprocess
import split_data
import svm_model

def run_custom_test():

    print("=== Running Custom Prediction Test ===")

    df = data_load.load_dataset("train.csv", sample_size=5000)
    X, y, scaler = preprocess.preprocess_data(df, target_col="Sex")
    X_train, _, y_train, _ = split_data.split_dataset(X, y)

    model = svm_model.train_svm(X_train, y_train, kernel_type='rbf')
    
    new_data = np.array([[30, 160.0, 55.0, 15.0, 95.0, 40.0, 80.0]])
    
    new_data_scaled = scaler.transform(new_data)
    
    prediction = model.predict(new_data_scaled)
    
    result = "Male" if prediction[0] == 1 else "Female"
    print("\n--- Prediction Result ---")
    print(f"Input Features : {new_data[0]}")
    print(f"Predicted Class: {result}")

if __name__ == "__main__":
    run_custom_test()