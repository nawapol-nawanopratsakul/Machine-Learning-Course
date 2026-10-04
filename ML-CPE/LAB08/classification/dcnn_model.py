from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def build_config_1(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=2, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def build_config_2(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(64, kernel_size=2, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, padding='same', activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def train_model(model, X_train, y_train, epochs, batch_size=32):
    print(f"Training model for {epochs} Epochs...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return model, history