from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel_type='linear', C=1.0, gamma='scale'):

    print(f"\nTraining SVM with '{kernel_type}' kernel (This might take a moment)...")
  
    model = SVC(kernel=kernel_type, C=C, gamma=gamma, random_state=42)
    

    model.fit(X_train, y_train)
    
    print(f"Training for '{kernel_type}' kernel complete.")
    return model