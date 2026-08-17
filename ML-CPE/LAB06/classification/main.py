import data_loader
import preprocessing
import split_data
import nn_model
import evaluate
import os
import numpy as np
import joblib
import json

def main():
    print("=== Start Neural Network Pipeline ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. โหลดข้อมูล
    df = data_loader.load_dataset("train.csv", sample_size=5000)
    
    # 2. ทำความสะอาดและสเกลข้อมูล
    X, y, scaler = preprocessing.preprocess_data(df, target_col="Sex")
    
    # 3. แบ่งข้อมูล Train / Test
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    
    # 4. บันทึก Dataset, Json, Scaler
    print("\nSaving datasets, classes, and scaler to 'outputs/' folder...")
    np.save(os.path.join(output_dir, "features.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    classes_dict = {"0": "female", "1": "male"}
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(classes_dict, f, indent=4)
        
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # 5. เทรนโมเดล Neural Network
    model = nn_model.train_nn(X_train, y_train, hidden_layer_sizes=(64, 32), max_iter=500)
    
    # 6. ประเมินผล
    acc = evaluate.evaluate_model(model, X_test, y_test, model_name="nn", output_dir="outputs")
    
    # 7. บันทึกโมเดล NN
    print("\nSaving the trained NN model...")
    joblib.dump(model, os.path.join(output_dir, "nn_model.pkl"))
    
    # สรุปผล
    print("\n" + "="*40)
    print(f"Neural Network Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("="*40)
    print("All production files are successfully saved in the 'outputs' folder!")

if __name__ == "__main__":
    main()