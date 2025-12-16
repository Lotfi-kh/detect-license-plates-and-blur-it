from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("🚀 Starting Training Engine...")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Found: {gpu_name} ")
    else:
        print("❌ WARNING: GPU not found! Training will be very slow.")

    # 1. Load Standard YOLOv11 (Small)
    model = YOLO("yolo11s.pt")

    # 2. Train
    results = model.train(
        data="dataset/data.yaml",  # Path to your config
        epochs=100,  # Max rounds
        patience=15,  # Stop early if no progress
        # --- 1660 Ti Specific Settings ---
        imgsz=1024,  # High resolution for better plate reading
        batch=4,  # Low batch size to prevent "Out of Memory"
        workers=2,  # Low workers to prevent Windows errors
        device=0,  # Force GPU usage
        project="yolo_plates",  # Main folder name
        name="run1",  # Sub-folder name
        exist_ok=True,  # Overwrite if exists
    )

    print("\n✅ Training Complete!")
    print("best.pt is located at: yolo_plates/run1/weights/best.pt")
