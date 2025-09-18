#!/usr/bin/env python3
"""
Simple YOLO training script for chessboard detection.
"""

from ultralytics import YOLO
import yaml

def train_yolo_chessboard():
    """Train YOLO model for chessboard detection"""
    print("🚀 Training YOLO Chessboard Detection")
    
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model
    results = model.train(
        data='yolo_chessboard_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolo_chessboard_v1',
        project='yolo_training_runs',
        save=True,
        plots=True,
        val=True,
        patience=15,
        device='cpu',
        workers=2
    )
    
    print("✅ YOLO training completed!")
    print(f"Best model: yolo_training_runs/yolo_chessboard_v1/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_yolo_chessboard()
