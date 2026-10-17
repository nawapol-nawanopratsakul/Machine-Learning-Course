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