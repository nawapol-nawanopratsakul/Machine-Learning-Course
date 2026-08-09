# LAB 04 - Calories Burned Dataset: Classification และ Clustering

โปรเจกต์นี้เป็นการทดลอง Machine Learning บนชุดข้อมูล Calories Burned Dataset (`train.csv`) เพื่อแสดงให้เห็นถึง 2 รูปแบบการเรียนรู้หลัก คือ

1. **Classification** ด้วยเทคนิค K-Nearest Neighbors (KNN) เพื่อจำแนกเพศ (Sex)
2. **Clustering** ด้วยเทคนิค K-Means ร่วมกับ TensorFlow เพื่อจัดกลุ่มข้อมูลสรีระและการเผาผลาญ

---

## 1. โครงสร้างโปรเจกต์

- **classification/**: โค้ดสำหรับงาน Classification
  - `main.py`: ไฟล์รันหลักสำหรับค้นหาค่า k และทดสอบโมเดล
  - `data_loader.py`: โหลดข้อมูล `train.csv` สุ่มตัวอย่าง 5,000 แถว และทำ Standardization
  - `knn_tf.py`: สร้างโมเดล KNN ด้วย TensorFlow
  - `evaluate.py`: สร้างกราฟ K-Curve, Confusion Matrix และบันทึกผลลัพธ์
  - `outputs/`: โฟลเดอร์เก็บรูปภาพกราฟและไฟล์ CSV ผลลัพธ์

- **clustering/**: โค้ดสำหรับงาน Clustering
  - `main.py`: ไฟล์รันหลักสำหรับประมวลผล K-Means ทั้งหมด
  - `data_loader.py`: โหลดข้อมูลและสเกลข้อมูลเฉพาะตัวแปรตัวเลข
  - `kmeans_tf.py`: โมเดล K-Means บน TensorFlow
  - `knn_tools.py`: โมเดล KNN สำหรับทำนายกลุ่มให้ข้อมูลใหม่ (Cluster Assigner)
  - `visualize.py`: สร้างกราฟ Elbow และ Scatter plot การกระจายกลุ่ม
  - `outputs/`: โฟลเดอร์เก็บผลลัพธ์การจัดกลุ่ม

- **data/**: โฟลเดอร์เก็บไฟล์ข้อมูล
  - `train.csv`: ชุดข้อมูลการออกกำลังกายและอัตราการเผาผลาญพลังงาน

---

## 2. Dataset ที่ใช้

ไฟล์ข้อมูลอยู่ที่:
- `data/train.csv`

คุณลักษณะข้อมูล:
- **Target (สำหรับ Classification):** คอลัมน์ `Sex` (female / male)
- **Features:** ตัวแปรเชิงปริมาณ ได้แก่ `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, และ `Calories`
- **Sampling:** สุ่มข้อมูลจำนวน 5,000 แถวมาใช้ในการทดลอง เพื่อลดระยะเวลาการคำนวณระยะทางของ TensorFlow

---

## 3. ส่วนที่ 1: Classification ด้วย KNN

### วัตถุประสงค์
เพื่อจำแนกเพศ (`Sex`) จากข้อมูลสรีระและการออกกำลังกาย โดยค้นหาค่า **k** ที่เหมาะสมที่สุด และเปรียบเทียบผลลัพธ์ระหว่างโมเดลที่สร้างเองบน TensorFlow กับ `scikit-learn`

### กระบวนการทำงาน
1. โหลดข้อมูล `train.csv` ผ่าน `data_loader.py` และแปลงค่าคลาส `Sex` ให้เป็นตัวเลข (0, 1)
2. แบ่งข้อมูลเป็น Train (60%), Validation (20%) และ Test (20%)
3. ปรับมาตราส่วนข้อมูลด้วย `StandardScaler`
4. ทดสอบหาค่า **k** ที่ดีที่สุดบน Validation Set ผ่านค่า **k** = 1, 2, 3, 5, 10, 11, 15, 21, 30
5. นำค่า **k** ที่ดีที่สุดไปทดสอบบน Test Set พร้อมวัดผลด้วย Accuracy, Classification Report และ Confusion Matrix
6. เปรียบเทียบผลลัพธ์กับ Baseline (การสุ่มทาย) และโมเดลมาตรฐานของ `scikit-learn`

### ผลลัพธ์ที่ได้ (`classification/outputs/`)
- `01_k_curve.png`: กราฟแสดงความสัมพันธ์ระหว่างค่า **k** กับ Validation Accuracy
- `02_confusion_matrix.png`: เมทริกซ์แสดงจำนวนการจำแนกเพศที่ถูกต้องและผิดพลาด
- `predictions.csv`: ตารางบันทึกค่าจริง (`true_label`) และค่าที่โมเดลทาย (`predicted_label`)

### วิธีรัน
```bash
python classification/main.py

---

## 4. ส่วนที่ 2: Clustering ด้วย K-Means

### วัตถุประสงค์
-  เพื่อจัดกลุ่มผู้ใช้ออกกำลังกายตามความคล้ายคลึงของสรีระและการเผาผลาญโดยไม่ใช้ Label เฉลย และใช้ KNN ในการทำนายกลุ่มให้กับข้อมูลใหม่

### กระบวนการทำงาน
1. โหลดข้อมูลตัวเลขทั้งหมดและทำ Standardization ด้วย StandardScaler
2. Elbow Method คำนวณค่า Inertia สำหรับจำนวนกลุ่มตั้งแต่ 2 ถึง 10 เพื่อหาจุดศอก
3. TFKMeans ด้วยจำนวนกลุ่มที่เหมาะสม (สมมติว่าเป็น 2 กลุ่ม)
4. Scatter Plot แสดงการกระจายตัวของกลุ่มโดยอ้างอิงจากแกน Weight และ Height
5. บันทึกผลลัพธ์การจัดกลุ่มลงไฟล์ CSV และสรุปค่าเฉลี่ยของแต่ละกลุ่ม
6. ทดสอบโมเดล KNNClusterAssigner เพื่อนำจุดข้อมูลใหม่ไปค้นหากลุ่มเพื่อนบ้านที่ใกล้ที่สุด

### ผลลัพธ์ที่ได้ (clustering/outputs/)
- 01_elbow.png: กราฟ Elbow Method ช่วยตัดสินใจเลือกจำนวน Cluster
- 02_clusters.png: ภาพการกระจายตัวของกลุ่มข้อมูลบนสเปซสรีระ
- clustered_data.csv: ข้อมูลต้นฉบับพร้อมคอลัมน์ Cluster แสดงกลุ่มที่สังกัด
- cluster_summary.csv: ค่าเฉลี่ยของคุณลักษณะต่างๆ แยกตาม Cluster

### วิธีรัน
```bash
python clustering/main.py

## 5. คำอธิบายสั้น ๆ ของเทคนิคที่ใช้
- KNN (K-Nearest Neighbors):
อัลกอริทึมจำแนกประเภทข้อมูลโดยคำนวณระยะทาง Euclidean จากจุดข้อมูลใหม่ไปยังจุดข้อมูลฝึกสอนทั้งหมด แล้วทำการโหวตเสียงข้างมาก (Majority Voting) จากเพื่อนบ้านที่ใกล้ที่สุด k ตัว

- K-Means Clustering:
อัลกอริทึมจัดกลุ่มแบบไร้ผู้สอน โดยกำหนดจุดศูนย์กลาง (Centroids) และวนรอบปรับตำแหน่งจนกระทั่งระยะห่างระหว่างจุดข้อมูลในกลุ่มเดียวกันมีความกระชับที่สุด (Inertia ต่ำสุด)

## 6. สรุป
โปรเจกต์นี้แสดงให้เห็นถึงการประยุกต์ใช้ Machine Learning สองรูปแบบกับชุดข้อมูล Calories Burned Dataset โดย Classification ช่วยให้เราเข้าใจการใช้ KNN ทายเพศจากลักษณะสรีระ และ Clustering ช่วยจัดกลุ่มผู้ใช้ออกกำลังกายที่มีลักษณะสรีระใกล้เคียงกันโดยไม่ต้องพึ่งพา Label เฉลย ทำให้เข้าใจกระบวนการทำงานพื้นฐานของทั้ง Supervised และ Unsupervised Learning ได้อย่างชัดเจน