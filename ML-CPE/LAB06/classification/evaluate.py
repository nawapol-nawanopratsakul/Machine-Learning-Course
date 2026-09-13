from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name="nn", output_dir="outputs"):
    """
    ประเมินผลความแม่นยำและสร้างกราฟ Confusion Matrix
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(current_dir, output_dir)
    
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['female', 'male']))
    
    # วาดรูป Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['female', 'male'], yticklabels=['female', 'male'])
    plt.title('Confusion Matrix (Neural Network)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # เซฟรูปลงโฟลเดอร์ outputs
    file_path = os.path.join(full_output_dir, f'cm_{model_name}.png')
    plt.savefig(file_path, dpi=120)
    plt.close()
    
    return acc