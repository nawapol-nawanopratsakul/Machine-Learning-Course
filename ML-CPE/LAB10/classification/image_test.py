from ultralytics import YOLO


def test_image():
  model = YOLO("runs/car_detection_lab/weights/best.pt")

  # ทำนายผลบนภาพตัวอย่าง
  results = model("test_car.jpg", show=True, conf=0.5)
  print("Object detection on image completed!")


if __name__ == "__main__":
  test_image()