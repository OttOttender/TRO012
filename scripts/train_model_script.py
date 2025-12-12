import os
import yaml
from ultralytics import YOLO
from roboflow import Roboflow

ROBOFLOW_API_KEY = "M4GM7XoEddkHym146c4B"
ROBOFLOW_WORKSPACE = "tro012"
ROBOFLOW_PROJECT = "battery-detection-bscuv"
ROBOFLOW_VERSION = 1

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_SIZE = 'n'
OUTPUT_DIR = "models"

def fix_data_yaml(data_path):
    """Fixes absolute paths and handles missing validation sets."""
    yaml_path = os.path.join(data_path, 'data.yaml')
    abs_path = os.path.abspath(data_path).replace("\\", "/")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    train_path = f"{abs_path}/train/images"
    valid_path = f"{abs_path}/valid/images"
    
    data['train'] = train_path
    data['val'] = valid_path if os.path.exists(valid_path) else train_path
    data['test'] = f"{abs_path}/test/images"
    
    if 'path' in data:
        del data['path']

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Config patched. Validation set: {'Found' if os.path.exists(valid_path) else 'Missing (using train set)'}.")

def main():
    print("Downloading dataset...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    
    fix_data_yaml(dataset.location)
    
    print("Starting training...")
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')
    
    model.train(
        data=os.path.join(dataset.location, 'data.yaml'),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='battery_detector',
        project=OUTPUT_DIR,
        patience=10,
        save=True,
        plots=True,
        exist_ok=True
    )
    
    print(f"Training complete. Best model: {OUTPUT_DIR}/battery_detector/weights/best.pt")

if __name__ == "__main__":
    main()