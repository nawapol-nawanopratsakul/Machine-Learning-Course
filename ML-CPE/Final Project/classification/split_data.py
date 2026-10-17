from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Reshape ข้อมูลให้เข้ากับโครงสร้าง 1D CNN (Samples, Features, Channels)
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(
        f"แบ่งข้อมูลสำเร็จ -> Train: {X_train_cnn.shape}, Test: {X_test_cnn.shape}"
    )
    return X_train_cnn, X_test_cnn, y_train, y_test