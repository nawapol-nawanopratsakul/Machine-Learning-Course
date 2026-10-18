# Used Car Price Prediction (Miniproject)

ระบบทำนายราคารถยนต์มือสองโดยใช้โมเดล Deep Learning (1D CNN Regression) ประมวลผลจากข้อมูลตาราง (Tabular Data) ร่วมกับการปรับเทียบค่าตัวคูณราคาให้เหมาะสมกับตลาดรถยนต์มือสองในประเทศไทย พร้อมโครงสร้างการเชื่อมต่อผ่าน RESTful API ด้วย Node.js

เว็บอ้างอิง
https://www.kaggle.com/datasets/rishabhkarn/used-car-dataset
https://www.kaggle.com/code/mohaiminul101/car-price-prediction/input

---

## 📋 คุณลักษณะของระบบ (Features)

1. **Data Preprocessing & Cleaning:** 
   - รวมข้อมูลจากหลายชุดข้อมูล (`Car details.csv`, `car data.csv`, `Used Car Dataset.csv`) เข้าด้วยกันเป็น `combined_cars.csv` (ประมาณ 4,708 รายการ)
   - สกัดและแปลงข้อมูลปีด้วย Regular Expression (Regex) พร้อมคำนวณอายุรถเทียบกับปีปัจจุบัน
   - แปลงหน่วยสกุลเงินและปรับขนาดข้อมูลด้วย `StandardScaler` / `LabelEncoder`

2. **1D CNN Regression Model:**
   - สถาปัตยกรรม 1D Convolutional Neural Network สำหรับประมวลผลข้อมูลตาราง
   - ปรับแต่ง Layer และ Loss Function (MSE / MAE) ให้เหมาะกับการทำนายค่าตัวเลขต่อเนื่อง

3. **Thai Market Price Calibration:**
   - ปรับแต่งผลลัพธ์ผ่านสคริปต์ทำนายราคา (`predict.py`) เพื่อประเมินราคาซื้อขายจริงในตลาดไทยอย่างสมเหตุสมผล

4. **Node.js Integration:**
   - รองรับการเรียกใช้งานผ่าน Node.js Backend API (Child Process / REST API) เพื่อส่งค่าอินพุตจากหน้าเว็บไปยังสคริปต์ Python และคืนผลลัพธ์แบบ Real-time

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
car-price-prediction/
├── data/
│   ├── raw/                  # ชุดข้อมูลดิบ
│   └── combined_cars.csv     # ชุดข้อมูลรวมที่ทำความสะอาดแล้ว
├── models/
│   ├── car_price_model.h5    # ไฟล์โมเดลที่ฝึกสอนเรียบร้อยแล้ว
│   └── scaler.pkl            # ไฟล์บันทึกการปรับสเกลข้อมูล
├── preprocess.py             # สคริปต์จัดการข้อมูลดิบ
├── main.py                   # สคริปต์ฝึกสอนและประเมินผลโมเดล (1D CNN)
├── predict.py                # สคริปต์รับพารามิเตอร์และทำนายราคา
├── server.js                 # Node.js Express Server (Backend API)
├── package.json              # ไฟล์การตั้งค่า Node.js
├── requirements.txt          # รายชื่อไลบรารี Python ที่ต้องติดตั้ง
└── README.md                 # เอกสารอธิบายโปรเจกต์
```

---

## 🛠️ การติดตั้งและเตรียมสภาพแวดล้อม (Setup & Installation)

### 1. การเตรียม Python Environment

ติดตั้งไลบรารี Python ที่จำเป็น:

```bash
pip install tensorflow numpy pandas scikit-learn joblib
```

### 2. การเตรียม Node.js Environment

ติดตั้ง Dependencies สำหรับ Node.js API:

```bash
npm install express cors body-parser
```

---

## 🚀 ขั้นตอนการรันระบบ (Workflow)

### ขั้นตอนที่ 1: เตรียมและทำความสะอาดข้อมูล
```bash
python preprocess.py
```

### ขั้นตอนที่ 2: ฝึกสอนโมเดล (Train Model)
```bash
python main.py
```
*(เมื่อรันเสร็จเรียบร้อย จะได้ไฟล์ `models/car_price_model.h5` และไฟล์ Encoders/Scalers ที่จำเป็น)*

### ขั้นตอนที่ 3: ทดสอบการทำนายราคาผ่าน Python
```bash
python predict.py --brand Toyota --year 2018 --km 50000 --fuel Petrol
```

### ขั้นตอนที่ 4: รัน Backend Server (Node.js)
```bash
node server.js
```

---

## 📊 การประเมินประสิทธิภาพ (Model Evaluation)

ประเมินผลลัพธ์โมเดลด้วยดัชนีชี้วัดมาตรฐานงาน Regression:
* **MAE (Mean Absolute Error):** ความคลาดเคลื่อนเฉลี่ยของราคารถยนต์
* **RMSE (Root Mean Squared Error):** ค่าความคลาดเคลื่อนกำลังสองเฉลี่ย
* **$R^2$ Score:** ค่าสัมประสิทธิ์การตัดสินใจแสดงความสามารถในการทำนายของโมเดล