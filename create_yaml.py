import yaml
import os

# ================= CONFIGURATION =================
# Point this to the folder containing 'classes.txt', 'images', and 'labels'
data_folder = r"C:\Users\asusr\OneDrive\Desktop\for_coding\yolo\test2\dataset\data"
# =================================================


def generate_yaml():
    # 1. Read classes.txt
    classes_path = os.path.join(data_folder, "classes.txt")

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            # Read lines and remove empty space
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Found classes.txt! Detected classes: {class_names}")

    # 2. Define the data structure for YOLO
    yolo_config = {
        "path": data_folder,  # Absolute path to your data root
        "train": "images/train",  # Relative path to train images
        "val": "images/val",  # Relative path to val images
        # This maps ID 0 to the first name in your file, ID 1 to the second, etc.
        "names": {i: name for i, name in enumerate(class_names)},
    }

    # 3. Save as data.yaml (saving it one folder up, in 'dataset/')
    output_path = os.path.join(data_folder, "..", "data.yaml")

    with open(output_path, "w") as f:
        yaml.dump(yolo_config, f, sort_keys=False)

    print("SUCCESS")


if __name__ == "__main__":
    generate_yaml()
