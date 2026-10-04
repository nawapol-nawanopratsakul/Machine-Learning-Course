import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_single_history(history, model_name, epochs):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'Accuracy: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'Loss: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/history_{model_name}_{epochs}Epochs.png")
    plt.close()

def plot_paper_style_history(history_1, history_2, model_names):
    os.makedirs("outputs", exist_ok=True)
    sns.set_style("darkgrid")
    plt.rcParams['axes.facecolor'] = '#EAEAF2'
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    
    def plot_panel(ax, metric, title, y_label):
        epochs_1 = range(1, len(history_1.history[metric]) + 1)
        epochs_2 = range(1, len(history_2.history[metric]) + 1)
        
        ax.plot(epochs_1, history_1.history[metric], marker='o', markersize=4, linestyle='--', color='royalblue', label=model_names[0])
        ax.plot(epochs_2, history_2.history[metric], marker='s', markersize=4, linestyle='-', color='mediumvioletred', label=model_names[1])
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.legend(loc='lower right' if 'loss' not in metric else 'upper right', fontsize=9)

    plot_panel(axs[0, 0], 'accuracy', '(a) Training accuracy of models.', 'Accuracy (%)')
    plot_panel(axs[0, 1], 'val_accuracy', '(b) Validation accuracy of models.', 'Validation Accuracy (%)')
    plot_panel(axs[1, 0], 'loss', '(c) Training loss of models.', 'Loss')
    plot_panel(axs[1, 1], 'val_loss', '(d) Validation loss of model.', 'Loss')
    plot_panel(axs[2, 0], 'precision', '(e) Precision performance of training model.', 'Precision (%)')
    plot_panel(axs[2, 1], 'recall', '(f) Recall performance of training model.', 'Recall (%)')

    plt.tight_layout()
    plt.savefig("outputs/research_paper_graphs.png", dpi=300)
    plt.close()

def evaluate_model(model, X_test, y_test, save_cm=False):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    if save_cm:
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (DCNN)')
        plt.savefig("outputs/cm_cnn.png")
        plt.close()
        
    return accuracy_score(y_test, y_pred)