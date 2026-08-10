import data_load
import preprocess
import split_data
import svm_model
import evaluate
import os
import numpy as np
import joblib
import json

def main():
    print("=== Start SVM Machine Learning Pipeline ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = data_load.load_dataset("train.csv", sample_size=5000)
    
    X, y, scaler = preprocess.preprocess_data(df, target_col="Sex")
    
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    
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
    # ---------------------------------------------------------

    kernels = ['linear', 'poly', 'rbf']
    accuracies = {}
    
    best_model = None
    best_acc = 0
    
    for kernel in kernels:
        model = svm_model.train_svm(X_train, y_train, kernel_type=kernel)
        
        acc = evaluate.evaluate_model(model, X_test, y_test, kernel_name=kernel, output_dir="outputs")
        accuracies[kernel] = acc
        

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\nSaving the best SVM model...")
    joblib.dump(best_model, os.path.join(output_dir, "svm_model.pkl"))
    
    print("\n" + "="*40)
    print("       SUMMARY OF SVM KERNELS")
    print("="*40)
    for k, v in accuracies.items():
        print(f"Kernel: {k.upper():10} | Accuracy: {v:.4f} ({v*100:.2f}%)")
    print("="*40)
    print("All production files are successfully saved in the 'outputs' folder!")

if __name__ == "__main__":
    main()