X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"จำนวนข้อมูลสำหรับ Train: {X_train.shape[0]} แถว")
print(f"จำนวนข้อมูลสำหรับ Test: {X_test.shape[0]} แถว")