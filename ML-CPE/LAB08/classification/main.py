import os
import joblib
import data_loader, preprocessing, split_data, dcnn_model, evaluate

def main():
    print("============================================================")
    print("LAB 08: Deep Convolutional Neural Network (DCNN)")
    print("============================================================")
    
    os.makedirs("outputs", exist_ok=True)
    
    df = data_loader.load_dataset("train.csv") 
    X, y, scaler = preprocessing.preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data.split_dataset(X, y)
    input_shape = (X_train.shape[1], 1)
    
    EPOCHS_1 = 100
    print(f"\n[Config 1] Standard CNN (1 Conv Layer) | {EPOCHS_1} Epochs")
    model_1 = dcnn_model.build_config_1(input_shape)
    model_1, history_1 = dcnn_model.train_model(model_1, X_train, y_train, epochs=EPOCHS_1)
    evaluate.plot_single_history(history_1, "Model_1", EPOCHS_1)
    acc_1 = evaluate.evaluate_model(model_1, X_test, y_test, save_cm=False)
    
    EPOCHS_2 = 200
    print(f"\n[Config 2] Deep CNN (4 Conv Layers + BatchNorm) | {EPOCHS_2} Epochs")
    model_2 = dcnn_model.build_config_2(input_shape)
    model_2, history_2 = dcnn_model.train_model(model_2, X_train, y_train, epochs=EPOCHS_2)
    evaluate.plot_single_history(history_2, "Model_2", EPOCHS_2)
    acc_2 = evaluate.evaluate_model(model_2, X_test, y_test, save_cm=True)
    
    labels = [f"Shallow ({EPOCHS_1} ep)", f"Deep ({EPOCHS_2} ep)"]
    evaluate.plot_paper_style_history(history_1, history_2, labels)
    
    joblib.dump(scaler, "outputs/scaler.pkl")
    model_2.save("outputs/cnn_model.keras")
    
    print("\n========================================")
    print("Performance Comparison")
    print("========================================")
    print(f"Config 1 (Shallow CNN, {EPOCHS_1} Epochs) Accuracy : {acc_1 * 100:.2f}%")
    print(f"Config 2 (Deep CNN, {EPOCHS_2} Epochs) Accuracy    : {acc_2 * 100:.2f}%")
    
    print("\nComplete! Check the 'outputs/' folder for the 6 exact files requested.")

if __name__ == "__main__":
    main()