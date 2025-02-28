from ultralytics import YOLO
import os
import torch  # ✅ Import PyTorch

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Use "yolov8m.pt" for better accuracy

# Train model (using data.yaml)
model.train(
    data="D:\Projects\FoodCalorieEstimator\data.yaml",  # Use YAML file instead of a dictionary
    epochs=50,         # Adjust epochs as needed
    imgsz=640,         # Image size for training
    batch=16,          # Batch size
    workers=2,         # Number of workers for data loading
    optimizer="Adam",
    lr0=0.001,         # Initial learning rate
    device="cuda" if torch.cuda.is_available() else "cpu"  # ✅ Use GPU if available
)

# Print confirmation
print("YOLOv8 training started successfully!")
