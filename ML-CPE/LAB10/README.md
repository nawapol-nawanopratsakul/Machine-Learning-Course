# LAB 10: Object Detection II (YOLO) - Machine Learning

เอกสารและโค้ดตัวอย่างสำหรับการทดลองในปฏิบัติการรายวิชา Machine Learning เรื่อง การตรวจจับวัตถุด้วยอัลกอริทึม YOLO (You Only Look Once) โดยใช้ไลบรารี Ultralytics YOLOv8

---

## 📋 วัตถุประสงค์ (Objectives)

1. เพื่อให้นักศึกษาเข้าใจหลักการทำงานของ Object Detection (การจำแนกประเภทพร้อมระบุตำแหน่ง bounding box ของวัตถุในภาพ)
2. เพื่อให้สามารถฝึกสอน (Train), ประเมินประสิทธิภาพ (Evaluate) และปรับแต่งค่าพารามิเตอร์ของโมเดล YOLO ได้
3. สามารถนำโมเดลไปประยุกต์ใช้ตรวจจับวัตถุจากรูปภาพ, วิดีโอ และกล้องแบบ Real-time

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB10/
├── train.py          # สคริปต์สำหรับฝึกสอนโมเดล (Training)
├── evaluate.py       # สคริปต์สำหรับประเมินผลโมเดล (Metrics: mAP, Precision, Recall)
├── image_test.py     # สคริปต์ทดสอบตรวจจับวัตถุจากรูปภาพ
├── realtime_test.py  # สคริปต์ทดสอบตรวจจับวัตถุผ่านกล้อง Webcam แบบ Real-time
└── requirements.txt  # รายชื่อแพ็กเกจที่จำเป็นสำหรับโปรเจกต์
```

---

## 🛠️ การติดตั้งและเตรียมความพร้อม (Installation)

1. ติดตั้งไลบรารีที่จำเป็นผ่านคำสั่ง pip:

```bash
pip install ultralytics opencv-python torch torchvision matplotlib pandas
```

---

## 🚀 วิธีการใช้งานสคริปต์ (Usage)

### 1. การฝึกสอนโมเดล (Training)

รันสคริปต์เพื่อเริ่มเทรนโมเดล YOLOv8 (สามารถใช้ชุดข้อมูลมาตรฐาน `coco8.yaml` หรือชุดข้อมูล Custom Dataset ของตัวเอง):

```bash
python train.py
```

**ตัวอย่างการตั้งค่าในโค้ด:**
- **Model:** `yolov8n.pt` (YOLOv8 Nano)
- **Epochs:** 10
- **Image Size:** 480x480
- **Batch Size:** 8
- **Device:** CPU / GPU

### 2. การประเมินผลโมเดล (Evaluation)

ตรวจสอบค่าความแม่นยำและประสิทธิภาพของโมเดลหลังจากเทรนเสร็จ:

```bash
python evaluate.py
```

*(แสดงผลค่า Precision, Recall, และ mAP)*

### 3. การทดสอบกับรูปภาพ (Image Detection)

ทดสอบการตรวจจับวัตถุบนภาพถ่าย:

```bash
python image_test.py
```

### 4. การทดสอบแบบ Real-time

เปิดกล้อง Webcam เพื่อทดสอบการตรวจจับวัตถุแบบเรียลไทม์:

```bash
python realtime_test.py
```

---

## 📊 ผลลัพธ์และชิ้นงาน (Output)

- โฟลเดอร์ `runs/` จะเก็บไฟล์น้ำหนักของโมเดล (`best.pt`, `last.pt`), กราฟ Loss และผลลัพธ์การประเมินผล (Training CSV / Validation Plots) ตามโจทย์ข้อกำหนดของปฏิบัติการ