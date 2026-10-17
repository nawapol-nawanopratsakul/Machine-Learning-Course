from pathlib import Path
from ultralytics import YOLO


def main():
  # 1. โหลดพรีไทรนโมเดล YOLOv8 Nano
  model = YOLO("yolov8n.pt")

  # 2. เริ่มเทรนโมเดลตามโจทย์ Lab
  results = model.train(
      data="coco8.yaml",  # พาธไฟล์กำหนดค่า Dataset
      epochs=10,  # จำนวนรอบในการเทรน (สามารถปรับเปลี่ยนเพื่อเปรียบเทียบได้)
      imgsz=480,  # ขนาดรูปภาพ
      batch=8,  # ขนาด Batch size
      device="cpu",  # ใช้ GPU (หรือใส่ 'cpu' ถ้าไม่มีการ์ดจอ)
      workers=0,
      amp=True,
      cache=False,
      project="runs",
      name="car_detection_lab",
      exist_ok=True,
      save=True,
      plots=True,
  )

  # 3. แสดงตำแหน่งที่จัดเก็บผลลัพธ์
  save_dir = Path(model.trainer.save_dir)
  print("Training completed!")
  print(f"Best Model: {save_dir / 'weights/best.pt'}")
  print(f"Last Model: {save_dir / 'weights/last.pt'}")
  print(f"Training CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
  main()