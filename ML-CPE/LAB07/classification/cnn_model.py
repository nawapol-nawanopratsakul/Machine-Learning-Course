from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def build_model_config_1(input_shape):
    model = Sequential([
        Conv1D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # เพิ่ม Precision และ Recall เข้าไปในตัววัดผล
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def build_model_config_2(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def train_model(model, X_train, y_train, epochs, batch_size=32):
    print(f"กำลังเทรนโมเดล จำนวน {epochs} Epochs...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return model, history