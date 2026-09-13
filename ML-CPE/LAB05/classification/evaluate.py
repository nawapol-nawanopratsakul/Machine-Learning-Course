import os, matplotlib.pyplot as plt, seaborn as sns  # นำเข้าเครื่องมือจัดการไฟล์และวาดกราฟ
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    # สร้างโฟลเดอร์ 'outputs' มารอรับไฟล์รูปภาพ (ถ้ามีแล้วจะไม่แจ้ง Error)
    os.makedirs("outputs", exist_ok=True)
    
    # สั่งให้โมเดลทำนายข้อมูลชุด Test
    y_pred = model.predict(X_test)
    
    # พิมพ์ค่าความแม่นยำ (Accuracy) และรายงาน Precision/Recall ออกทางหน้าจอ
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n", classification_report(y_test, y_pred))
    
    # วาดกราฟตาราง Confusion Matrix สีฟ้า พร้อมใส่ตัวเลขกำกับ
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    
    # บันทึกรูปกราฟลงในโฟลเดอร์ outputs
    plt.savefig("outputs/cm_svm.png")
    
    # ล้างหน่วยความจำกราฟิกเพื่อไม่ให้รกเครื่อง
    plt.close()