from sklearn.svm import SVC  # นำเข้าคลาส Support Vector Classification

def train_svm(X_train, y_train, kernel_type='linear'):
    # สร้างโมเดล SVM ชนิด linear และสั่งให้เรียนรู้ (fit) ข้อมูลชุด Train ในบรรทัดเดียว
    return SVC(kernel=kernel_type, C=1.0, random_state=42).fit(X_train, y_train)