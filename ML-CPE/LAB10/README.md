# ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี
**รหัสวิชา:** 04-624-201 ชื่อวิชา: Machine Learning  
**ใบงานที่ 10:** Object Detection II (YOLO)

---

## 📋 วัตถุประสงค์ (Objectives)
1. เพื่อให้นักศึกษาเข้าใจหลักการทำงานของ Object Detection โดยใช้ YOLO (You Only Look Once) สำหรับการตรวจจับ จำแนกประเภท และระบุตำแหน่งวัตถุภายในภาพ
2. เพื่อให้นักศึกษาสามารถพัฒนา ฝึกสอน และประเมินประสิทธิภาพแบบจำลอง YOLO รวมถึงนำไปประยุกต์ใช้ในการตรวจจับวัตถุจากภาพ วิดีโอ และกล้องแบบเรียลไทม์

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
ติดตั้งไลบรารีที่จำเป็นผ่านคำสั่ง pip:
```bash
pip install ultralytics opencv-python torch torchvision matplotlib pandas
```

---

## 💻 ซอร์สโค้ดสำหรับแต่ละโมดูล (Source Code)

### 1. ไฟล์สำหรับเทรนโมเดล (`train.py`)
```python
from pathlib import Path
from ultralytics import YOLO


def main():
  # โหลดพรีไทรนโมเดล YOLOv8 Nano
  model = YOLO("yolov8n.pt")

  # เริ่มเทรนโมเดลตามโจทย์ Lab (ใช้ coco8.yaml สำหรับทดสอบความสะดวกรวดเร็ว)
  results = model.train(
      data="coco8.yaml",
      epochs=10,
      imgsz=480,
      batch=8,
      device="cpu",  # ใช้ CPU หากเครื่องไม่มีการ์ดจอ NVIDIA ที่รองรับ CUDA
      workers=0,
      amp=True,
      cache=False,
      project="runs",
      name="car_detection_lab",
      exist_ok=True,
      save=True,
      plots=True,
  )

  save_dir = Path(model.trainer.save_dir)
  print("Training completed!")
  print(f"Best Model: {save_dir / 'weights/best.pt'}")
  print(f"Last Model: {save_dir / 'weights/last.pt'}")
  print(f"Training CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
  main()
```

### 2. ไฟล์สำหรับประเมินผลโมเดล (`evaluate.py`)
```python
from ultralytics import YOLO


def evaluate_model():
  # โหลดโมเดลที่เทรนเสร็จแล้ว
  model = YOLO("runs/car_detection_lab/weights/best.pt")

  # ประเมินผลบนชุดทดสอบ (Validation/Test set)
  metrics = model.val()

  print(f"mAP50-95: {metrics.box.map}")
  print(f"mAP50: {metrics.box.map50}")
  print(f"Precision: {metrics.box.p}")
  print(f"Recall: {metrics.box.r}")


if __name__ == "__main__":
  evaluate_model()
```

### 3. ไฟล์สำหรับทดสอบกับรูปภาพ (`image_test.py`)
```python
from ultralytics import YOLO


def test_image():
  model = YOLO("runs/car_detection_lab/weights/best.pt")

  # ทำนายผลบนภาพตัวอย่าง
  results = model("test_car.jpg", show=True, conf=0.5)
  print("Object detection on image completed!")


if __name__ == "__main__":
  test_image()
```

### 4. ไฟล์สำหรับทดสอบแบบ Real-time (`realtime_test.py`)
```python
from ultralytics import YOLO


def realtime_detection():
  model = YOLO("runs/car_detection_lab/weights/best.pt")

  # เปิดกล้อง Webcam (0 คือกล้องหลัก)
  results = model(source=0, show=True, conf=0.4)


if __name__ == "__main__":
  realtime_detection()
```

### 5. รายชื่อแพ็กเกจ (`requirements.txt`)
```text
ultralytics>=8.0.0
opencv-python
torch
torchvision
matplotlib
pandas
```

---

## 📊 ผลลัพธ์และชิ้นงานที่คาดหวัง (Output)
* โฟลเดอร์ `runs/car_detection_lab/` จะเก็บไฟล์น้ำหนักของโมเดล (`weights/best.pt`), กราฟ Loss และผลลัพธ์การประเมินประสิทธิภาพ (Precision, Recall, mAP) พร้อมส่งรายงานตามโจทย์ปฏิบัติการ