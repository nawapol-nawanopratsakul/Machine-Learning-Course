from ultralytics import YOLO


def realtime_detection():
  model = YOLO("runs/car_detection_lab/weights/best.pt")

  # เปิดกล้อง Webcam (0 คือกล้องหลัก)
  results = model(source=0, show=True, conf=0.4)


if __name__ == "__main__":
  realtime_detection()