from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path("edge-ai-monitoring")
DATA_YAML = BASE_DIR / "data" / "processed" / "merged" / "data.yaml"

def train_model():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        name="edge_ai_model",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )

    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    train_model()




