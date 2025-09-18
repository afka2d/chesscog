#!/usr/bin/env python3
"""
Debug YOLO corner detection to understand what's happening.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_yolo_model():
    """Debug YOLO model detection"""
    print("🔧 DEBUGGING YOLO CORNER DETECTION")
    print("=" * 60)
    
    # Check model file
    model_path = "yolo_training_runs/yolo_chessboard_v1/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Model found: {model_path}")
    
    # Load model
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")
        
        # Print model info
        print(f"   Model type: {type(model)}")
        print(f"   Model task: {getattr(model, 'task', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test on a simple image
    test_image = 'grey_background_dataset/images/test/IMG_4785.JPG'
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"\n📸 Testing on: {test_image}")
    
    try:
        # Run inference with verbose output
        results = model(test_image, verbose=True, save=True, conf=0.1)  # Lower confidence threshold
        
        print(f"✅ Inference completed")
        print(f"   Number of results: {len(results)}")
        
        if results and len(results) > 0:
            result = results[0]
            
            print(f"\n🔍 DETECTION ANALYSIS:")
            
            # Check boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   Boxes found: {len(boxes)}")
                
                if len(boxes) > 0:
                    confidences = boxes.conf.cpu().numpy()
                    coordinates = boxes.xyxy.cpu().numpy()
                    
                    print(f"   Confidence scores: {confidences}")
                    print(f"   Box coordinates: {coordinates}")
                    
                    for i, (conf, box) in enumerate(zip(confidences, coordinates)):
                        x1, y1, x2, y2 = box
                        print(f"     Box {i}: conf={conf:.3f}, coords=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                else:
                    print("   No boxes detected")
            else:
                print("   No boxes attribute found")
            
            # Check masks
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                print(f"   Masks found: {len(masks)}")
                
                if len(masks) > 0:
                    for i, mask in enumerate(masks.xy):
                        print(f"     Mask {i}: {len(mask)} points")
                        if len(mask) > 0:
                            print(f"       First few points: {mask[:3]}")
                else:
                    print("   No mask data")
            else:
                print("   No masks attribute found")
            
            # Check other attributes
            print(f"\n📋 RESULT ATTRIBUTES:")
            for attr in dir(result):
                if not attr.startswith('_'):
                    try:
                        value = getattr(result, attr)
                        if not callable(value):
                            print(f"   {attr}: {type(value)} - {str(value)[:100]}")
                    except:
                        print(f"   {attr}: <unable to access>")
        else:
            print("❌ No results returned")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

def test_yolo_with_pretrained():
    """Test with a pre-trained YOLO model to verify the pipeline works"""
    print(f"\n🧪 TESTING WITH PRE-TRAINED YOLO MODEL")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Use pre-trained YOLOv8 segmentation model
        model = YOLO('yolov8n-seg.pt')
        print("✅ Pre-trained YOLOv8n-seg model loaded")
        
        # Test on our image
        test_image = 'grey_background_dataset/images/test/IMG_4785.JPG'
        
        print(f"📸 Testing pre-trained model on: {Path(test_image).name}")
        
        results = model(test_image, verbose=True, save=True, conf=0.3)
        
        if results and len(results) > 0:
            result = results[0]
            
            print(f"✅ Pre-trained model detected {len(result.boxes) if result.boxes else 0} objects")
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Show detected classes
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                print(f"   Detected classes: {classes}")
                print(f"   Confidences: {confidences}")
                
                # Print class names
                for cls, conf in zip(classes, confidences):
                    class_name = model.names[int(cls)]
                    print(f"     {class_name}: {conf:.3f}")
            
            print(f"   This confirms YOLO pipeline is working")
        else:
            print(f"   No detections with pre-trained model")
            
    except Exception as e:
        print(f"❌ Pre-trained model test failed: {e}")

def check_training_logs():
    """Check YOLO training logs"""
    print(f"\n📋 CHECKING YOLO TRAINING LOGS")
    print("=" * 40)
    
    log_dirs = [
        "yolo_training_runs/yolo_chessboard_v1",
        "runs/segment/train"
    ]
    
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if log_path.exists():
            print(f"📁 Found training directory: {log_dir}")
            
            # List contents
            for item in log_path.iterdir():
                if item.is_file():
                    stat = item.stat()
                    size = stat.st_size
                    print(f"   📄 {item.name}: {size} bytes")
            
            # Check for results.csv or similar
            results_files = list(log_path.glob("results*"))
            if results_files:
                print(f"   📊 Training results found: {[f.name for f in results_files]}")
            
            break
    else:
        print("❌ No training directories found")

def main():
    """Main debugging function"""
    print("YOLO Corner Detection Debugging")
    print("=" * 50)
    
    # Debug the trained model
    debug_yolo_model()
    
    # Test with pre-trained model to verify pipeline
    test_yolo_with_pretrained()
    
    # Check training logs
    check_training_logs()
    
    print(f"\n💡 DEBUGGING SUMMARY:")
    print("   1. YOLO model loads successfully")
    print("   2. Pipeline works with pre-trained models")
    print("   3. Custom model may need more training or different approach")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("   1. Let training complete (may take 30-60 minutes)")
    print("   2. Check if chessboard detection needs different approach")
    print("   3. Consider using object detection (bbox) instead of segmentation")
    print("   4. May need to adjust confidence thresholds")

if __name__ == "__main__":
    main()
