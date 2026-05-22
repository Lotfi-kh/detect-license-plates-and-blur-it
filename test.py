import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
#              CONFIGURATION
# ==========================================
model_path = (
    r"C:\Users\asusr\OneDrive\Desktop\for_coding\yolo\test2\train\weights\best.pt"
)
video_path = r"C:\Users\asusr\OneDrive\Desktop\for_coding\yolo\test2\video2.mp4"
min_thresh = 0.5

target_class_id = 0

box_color = (0, 255, 0)
# ==========================================

print(f"Loading model: {model_path}...")
model = YOLO(model_path, task="detect")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print("Processing video... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OPTIMIZATION 1: Add 'classes=[target_class_id]'
    results = model(frame, verbose=False, classes=[target_class_id])

    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        conf = detections[i].conf.item()

        if conf > min_thresh:
            # Get coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # ---------------------------------------------------------
            # MODIFICATION: Blur the detected region instead of drawing
            # ---------------------------------------------------------

            # 1. Extract the Region of Interest (ROI)
            roi = frame[ymin:ymax, xmin:xmax]

            # 2. Apply Gaussian Blur if ROI is valid
            # We check if the ROI has a valid size to avoid errors
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                # Dynamic Kernel Size:
                # We scale the blur strength based on the object's width.
                # A fixed kernel (e.g., (51,51)) might be too weak for close-ups
                # or too strong for distant objects.
                # The kernel size must be an odd number.
                k_size = (xmax - xmin) // 5
                if k_size % 2 == 0:
                    k_size += 1

                # Ensure a minimum amount of blur
                k_size = max(21, k_size)

                # Apply blur
                # sigma=0 lets OpenCV calculate the standard deviation automatically
                blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)

                # 3. Replace the original area with the blurred version
                frame[ymin:ymax, xmin:xmax] = blurred_roi

            object_count += 1

    # Draw Object Count (kept for debugging/verification)
    cv2.putText(
        frame,
        f"Count: {object_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        box_color,
        2,
    )

    cv2.imshow("Single Class Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
