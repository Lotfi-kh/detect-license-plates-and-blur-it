import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
#              CONFIGURATION
# ==========================================
model_path = (
    r"C:\Users\asusr\OneDrive\Desktop\for_coding\yolo\test2\train\weights\best.pt"
)
image_path = r"C:\Users\asusr\OneDrive\Desktop\for_coding\yolo\test2\test_image.jpeg"

min_thresh = 0.5
target_class_id = 0
# ==========================================

print(f"Loading model: {model_path}...")
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

        roi = frame[ymin:ymax, xmin:xmax]

        if roi.shape[0] > 0 and roi.shape[1] > 0:
            k_size = (xmax - xmin) // 5
            if k_size % 2 == 0:
                k_size += 1
            k_size = max(21, k_size)

            blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
            frame[ymin:ymax, xmin:xmax] = blurred_roi

        object_count += 1

cv2.putText(
    frame,
    f"Processed: {object_count} plates",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

# ---------------------------------------------------------
# DYNAMIC ASPECT RATIO FIX
# ---------------------------------------------------------
# 1. Get the original image dimensions
h, w = frame.shape[:2]

# 2. Define a safe height that fits on almost any laptop screen (e.g., 800 pixels)
target_height = 800

# 3. Calculate the correct width to maintain the ORIGINAL aspect ratio
#    Equation: (original_width / original_height) = (new_width / new_height)
aspect_ratio = w / h
target_width = int(target_height * aspect_ratio)

cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image Detection", target_width, target_height)

cv2.imshow("Image Detection", frame)
print("Image displayed. Press any key to close...")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = image_path.replace(".", "_blurred.")
cv2.imwrite(output_path, frame)
print(f"Saved result to: {output_path}")
