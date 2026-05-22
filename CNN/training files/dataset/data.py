from PIL import Image
import numpy as np


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_annotation(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Ensure the line has enough data
            if len(parts) >= 5:
                # Parse the values: Class Index (int) and Coordinates (float/int)
                cls_indx = int(parts[0])
                x0 = float(parts[1])
                y0 = float(parts[2])
                x1 = float(parts[3])
                y1 = float(parts[4])
                boxes.append((cls_indx, x0, y0, x1, y1))
    return boxes
