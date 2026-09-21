import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def plot_training_history(history, model_name):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/history_{model_name}.png")
    plt.close()
    print(f"บันทึกกราฟประวัติการเทรนไว้ที่ 'outputs/history_{model_name}.png'")

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    return acc