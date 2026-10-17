import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_cnn_model(input_shape):
    model = Sequential(
        [
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                padding="same",
                input_shape=input_shape,
            ),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(
                1, activation="linear"
            ),  #  Dense(1) สำหรับทำนายตัวเลขราคาตรงๆ
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mean_squared_error",  #  ใช้ Loss สำหรับ Regression
        metrics=["mae"],  #  วัดความคลาดเคลื่อนด้วย Mean Absolute Error
    )

    print("สร้างสถาปัตยกรรมโมเดล 1D CNN (Regression) สำเร็จ!")
    return model