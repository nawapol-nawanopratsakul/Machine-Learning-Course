from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def train_cnn(X_train, y_train, epochs=20, batch_size=32):
    print("\n=== เริ่มต้นฝึกสอนโมเดล 1D CNN ===")
    
    # โครงสร้าง CNN Architecture ประกอบด้วย 3 ส่วนหลัก: 
    # Convolution Layers, Pooling Layers และ Fully Connected Layers
    model = Sequential([
        # 1. Convolution Layer: ใช้ Filter เลื่อนผ่านข้อมูลเพื่อสกัด Feature ที่สำคัญ
        # กำหนด padding='valid' ซึ่งจะลดขนาด output ลงโดยไม่เติมพิกเซลพิเศษ
        Conv1D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=(X_train.shape[1], 1)),
        
        # 2. Pooling Layer: ใช้ Max Pooling ดึงเฉพาะค่าที่มากที่สุด (Largest value) 
        # เพื่อลดขนาด Feature map ทำให้เครือข่ายทำงานเร็วขึ้น
        MaxPooling1D(pool_size=2),
        
        # 3. Flatten Layer: แปลง Feature map แบบ 2 มิติ ให้กลายเป็นเวกเตอร์ 1 มิติ (1D vector) 
        # เพื่อเตรียมส่งเข้าสู่ชั้น Fully Connected
        Flatten(),
        
        # 4. Fully Connected Layer: ชั้นโครงข่ายประสาทเทียมแบบ Dense
        # ใช้ ReLU Activation Function เพื่อเพิ่ม Non-linearity ช่วยให้เรียนรู้รูปแบบที่ซับซ้อนได้
        Dense(16, activation='relu'),
        
        # 5. Output Layer: สำหรับ Classification
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return model