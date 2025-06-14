# src/train_yolo.py

from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11l.pt")  # pode usar yolov8s.pt, yolov8m.pt, etc.
    
    model.train(
        data=r"C:\Users\Bruno\Desktop\smart_security\data\annotations\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="models/yolo",
        name="yolo_furtos_v1",
        pretrained=True,
        verbose=True
    )

    # Avaliação
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    train_model()
