import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_paper_style_history(history_1, history_2, model_names):
    os.makedirs("outputs", exist_ok=True)
    
    sns.set_style("darkgrid")
    plt.rcParams['axes.facecolor'] = '#EAEAF2'
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    epochs = range(1, len(history_1.history['accuracy']) + 1)
    
    def plot_panel(ax, metric, title, y_label):
        ax.plot(epochs, history_1.history[metric], marker='o', markersize=4, linestyle='--', color='royalblue', label=model_names[0])
        ax.plot(epochs, history_2.history[metric], marker='s', markersize=4, linestyle='-', color='mediumvioletred', label=model_names[1])
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

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # วาดและเซฟรูป Confusion Matrix ของแต่ละโมเดล
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.savefig(f"outputs/cm_{model_name}.png")
    plt.close()
    
    return accuracy_score(y_test, y_pred)