from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path("edge-ai-monitoring")

def export_model():
    model_path = BASE_DIR / "runs" / "detect" / "edge_ai_model" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"Error: {model_path} does not exist.")
        print("Please ensure you have completed the training step (02_train_model.py).")
        return

    print(f"Loading trained model from {model_path}")
    model = YOLO(model_path)

    print("\n--- Exporting to ONNX ---")
    onnx_path = model.export(format="onnx", imgsz=640, half=True)
    print(f"ONNX model saved at: {onnx_path}")

    print("\n--- Exporting to TFLite (FP16) ---")
    tflite_path = model.export(format="tflite", imgsz=640, int8=False, half=True)
    print(f"TFLite model saved at: {tflite_path}")

if __name__ == "__main__":
    export_model()