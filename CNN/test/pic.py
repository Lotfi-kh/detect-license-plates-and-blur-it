import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
#              CONFIGURATION
# ==========================================
model_path = "train/weights/best.pt"
image_path = "a86.jpg"

min_thresh = 0.5
target_class_id = 0
# ==========================================

print("Loading model")
model = YOLO(model_path, task="detect")

frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not open image file {image_path}")
    exit()

print("Processing image...")

results = model(frame, verbose=False, classes=[target_class_id])
detections = results[0].boxes
object_count = 0

for i in range(len(detections)):
    conf = detections[i].conf.item()

    if conf > min_thresh:
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # 1. BLUR THE LICENSE PLATE (Existing logic)
        roi = frame[ymin:ymax, xmin:xmax]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            k_size = (xmax - xmin) // 5
            if k_size % 2 == 0:
                k_size += 1
            k_size = max(21, k_size)

            blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
            frame[ymin:ymax, xmin:xmax] = blurred_roi

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 3. DISPLAY PRECISION (Confidence Score)
        label = f"{conf:.2%}"  # Formats 0.853 as 85.30%

        # Calculate text size for the background box
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw filled rectangle for text background (for readability)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w_text, ymin), (0, 255, 0), -1)

        # Draw the text on top
        cv2.putText(
            frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        object_count += 1

# ---------------------------------------------------------
# DYNAMIC ASPECT RATIO FIX
# ---------------------------------------------------------
h, w = frame.shape[:2]
target_height = 600
aspect_ratio = w / h
target_width = int(target_height * aspect_ratio)

cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image Detection", target_width, target_height)

cv2.imshow("Image Detection", frame)
print(f"Processed {object_count} license plates. Press any key to close...")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = image_path.replace(".", "_detected.")
cv2.imwrite(output_path, frame)
print(f"Saved result to: {output_path}")
