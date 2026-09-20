# LAB - Convolutional Neural Network Classification (1D CNN)

โปรเจกต์นี้เป็นการทดลองพัฒนาระบบ Deep Learning ด้วยอัลกอริทึม Convolutional Neural Network (CNN) แบบ 1 มิติ (1D) เพื่อจำแนกประเภทข้อมูล (Classification) เพศชาย/หญิง จากข้อมูลทางสรีระและพฤติกรรมการออกกำลังกาย โดยปรับปรุงโครงสร้างโค้ดให้พร้อมสำหรับการนำไปใช้งานจริง (Production-ready) ด้วยการบันทึกสถานะของโมเดลและข้อมูลที่จำเป็น

Dataset ที่ใช้มาจาก Kaggle: https://www.kaggle.com/competitions/f-1-racer-diet-planning

---

### 1. โครงสร้างโปรเจกต์

ระบบถูกออกแบบให้ทำงานแบบแยกส่วน (Modular) เพื่อความสะดวกในการอ่าน จัดการ และนำไปต่อยอด:

*   `main.py` : ไฟล์หลักที่เรียกใช้งาน Pipeline ทั้งหมด ตั้งแต่ต้นจนถึงการบันทึกไฟล์
*   `data_loader.py` : โหลดชุดข้อมูลจากไฟล์ `train.csv` และจัดการพาธ (Path) แบบอัตโนมัติ
*   `preprocessing.py` : ทำความสะอาดข้อมูล แปลงคลาสเป้าหมายเป็นตัวเลข, ทำ Standardization และ **Reshape** ข้อมูลเป็น 3 มิติ
*   `split_data.py` : แบ่งข้อมูลออกเป็นชุดสำหรับฝึกสอน (Train) และชุดทดสอบ (Test)
*   `cnn_model.py` : สร้างและฝึกสอนโมเดล 1D Convolutional Neural Network (Conv1D)
*   `evaluate.py` : ประเมินความแม่นยำ (Accuracy) และบันทึกรูปภาพ Confusion Matrix
*   `test_cnn.py` : สคริปต์เสริมสำหรับโหลดโมเดลที่ฝึกสอนเสร็จแล้ว (Pre-trained) มาทำนายข้อมูลชุดใหม่
*   `outputs/` : โฟลเดอร์สำหรับจัดเก็บไฟล์ที่ระบบสร้างขึ้นทั้งหมด (ภาพกราฟ, ไฟล์ `.pkl`, `.keras`)
*   `train.csv` : ไฟล์ชุดข้อมูลต้นฉบับ (ต้องนำมาวางไว้ในโฟลเดอร์หลัก)

---

### 2. Dataset ที่ใช้

ไฟล์ข้อมูลที่เลือกใช้คือ `train.csv` (ข้อมูลอัตราการเผาผลาญแคลอรีและการออกกำลังกาย):

*   **Target:** คอลัมน์ `Sex` (จำแนกเพศ: `0 = female`, `1 = male`)
*   **Features:** ปัจจัยทางสรีระ ได้แก่ `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `Calories`
*   **Sampling:** ข้อมูลถูกสุ่มดึงมาใช้งานจำนวน 5,000 แถว เพื่อลดระยะเวลาในการประมวลผลของเครื่องคอมพิวเตอร์ แต่ยังคงประสิทธิภาพในการเรียนรู้ของโมเดลไว้

---

### 3. วัตถุประสงค์ (Objectives)

1. เพื่อเรียนรู้และประยุกต์ใช้อัลกอริทึม Deep Learning (Convolutional Neural Network) ในงาน Classification กับข้อมูลแบบตาราง (Tabular Data)
2. เพื่อทำความเข้าใจความสำคัญของการทำสเกลข้อมูล (Standardization) และการปรับรูปทรงข้อมูล (Reshape) ให้เข้ากับสถาปัตยกรรมของ CNN
3. เพื่อเรียนรู้วิธีการบันทึกโมเดล (Model Persistence) ทั้งในส่วนของ Scaler (`.pkl`) และ Deep Learning Model (`.keras`) เพื่อนำไปใช้งานซ้ำโดยไม่ต้องฝึกสอนใหม่

---

### 4. กระบวนการทำงาน (Workflow)

1. **Data Loading:** โหลดข้อมูล `train.csv` และลบคอลัมน์ที่ไม่ส่งผลต่อการทำนาย (`id`) ทิ้งไป
2. **Preprocessing:** แปลงข้อมูลเพศให้อยู่ในรูปตัวเลข (0 และ 1) จากนั้นแยก Features และ Target
3. **Standardization & Reshape:** ปรับสเกลของ Features ให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 ด้วย `StandardScaler` จากนั้นแปลงมิติข้อมูลจาก 2D เป็น 3D `(batch_size, 7, 1)` เพื่อให้เข้ากับ `Conv1D`
4. **Data Splitting:** แบ่งข้อมูลออกเป็น Train (80%) และ Test (20%) แบบ Stratified เพื่อรักษาสัดส่วนของเพศให้สมดุล
5. **Model Training:** ฝึกสอนโมเดล 1D CNN ด้วยโครงสร้าง Conv1D, MaxPooling1D, Flatten และ Dense (Fully Connected) เป็นจำนวน 20 Epochs
6. **Evaluation & Export:** ประเมินความแม่นยำ สร้างกราฟ Confusion Matrix และบันทึกไฟล์สเกลเลอร์พร้อมโมเดลที่เทรนเสร็จแล้วลงในโฟลเดอร์ `outputs/`

---

### 5. คำอธิบายสั้นๆ ของเทคนิคที่ใช้

*   **1D Convolutional Neural Network (Conv1D):** อัลกอริทึมที่ใช้แผ่นกรอง (Filter/Kernel) เลื่อนไปตามข้อมูลแบบลำดับ (Sequence) หรือข้อมูล 1 มิติ เพื่อสกัดคุณลักษณะ (Feature Extraction) ที่สำคัญออกมา
*   **Max Pooling (MaxPooling1D):** เทคนิคการลดขนาดของข้อมูล (Downsampling) โดยเลือกค่าที่มากที่สุดในแต่ละช่วงที่กำหนด เพื่อลดภาระการคำนวณและลดโอกาสเกิด Overfitting
*   **Flatten:** การแปลงข้อมูลที่มีหลายมิติ (จากชั้น Convolution และ Pooling) ให้แบนราบกลายเป็นมิติเดียว (1D Vector) เพื่อเตรียมส่งเข้าสู่ชั้น Fully Connected
*   **ReLU Activation Function:** ฟังก์ชันกระตุ้นที่โมเดลเลือกใช้ เพื่อช่วยให้โครงข่ายสามารถเรียนรู้รูปแบบที่ซับซ้อนได้อย่างรวดเร็วและแก้ปัญหาค่าความชันหายไป (Vanishing Gradient)
*   **Model Persistence (Joblib & Keras):** เทคนิคการแปลงออบเจกต์ในหน่วยความจำของ Python ให้อยู่ในรูปไฟล์ (เช่น `.pkl` สำหรับ Scaler และ `.keras` สำหรับ TensorFlow Model) เพื่อให้สามารถดึงกลับมาใช้งานทำนายข้อมูลใหม่ได้ทันที