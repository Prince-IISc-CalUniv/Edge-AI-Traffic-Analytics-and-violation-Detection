import cv2
import time
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path("edge-ai-monitoring")
MODEL_PATH_PT = BASE_DIR / "best.pt"
MODEL_PATH_ONNX = BASE_DIR / "runs" / "detect" / "edge_ai_model" / "weights" / "best.onnx"
MODEL_PATH_TFLITE = BASE_DIR / "best_int8.tflite"

def run_inference():
    if MODEL_PATH_PT.exists():
        model_to_load = MODEL_PATH_PT
    elif MODEL_PATH_ONNX.exists():
        model_to_load = MODEL_PATH_ONNX
    elif MODEL_PATH_TFLITE.exists():
        model_to_load = MODEL_PATH_TFLITE
    else:
        print("Model not found! Please place 'best.pt' in your main edge-ai-monitoring folder.")
        return

    print(f"Loading model from: {model_to_load}")
    try:
        model = YOLO(model_to_load)
    except Exception as e:
        print("Model not found. Please train and export the model first.")
        return

    print("\n--- Demo Setup ---")
    user_input = input("Enter '0' to use your Webcam, OR type the name of your video file (e.g. demo.mp4): ")

    if user_input.strip() == '0':
        video_source = 0
    else:
        video_source = user_input.strip()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting inference... Press 'q' to quit.")

    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.05, iou=0.5, verbose=False)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        annotated_frame = results[0].plot(line_width=4, font_size=1.5)

        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        hud_text = f"Edge AI Infrastructure Monitor | FPS: {fps:.1f} | Status: Active"
        cv2.putText(annotated_frame, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        height, width = annotated_frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            annotated_frame = cv2.resize(annotated_frame, (1280, int(height * scale)))

        cv2.imshow("Professional Edge AI Demo", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()