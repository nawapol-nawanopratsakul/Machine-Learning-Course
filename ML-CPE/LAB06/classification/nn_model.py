from sklearn.neural_network import MLPClassifier

def train_nn(X_train, y_train, hidden_layer_sizes=(64, 32), max_iter=500):
    """
    สร้างและฝึกสอน Neural Network (Multi-Layer Perceptron)
    """
    print(f"\nTraining Neural Network with hidden layers {hidden_layer_sizes}...")
    
    # สร้างโมเดล Neural Network โครงสร้าง 2 ชั้นซ่อน (64 โหนด และ 32 โหนด)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, 
        max_iter=max_iter, 
        activation='relu', # ใช้ ReLU เป็น activation function
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Neural Network training complete.")
    return model