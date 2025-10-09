#!/usr/bin/env python3
"""
Visualize corner detections from the new improved model.
Shows detected corners overlaid on test images.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import random

def visualize_corner_detections(model_path, num_samples=10):
    """Visualize corner detections from the model"""
    
    print("🔍 Loading improved corner detection model...")
    model = YOLO(model_path)
    
    # Get test images from the dataset
    test_images_dir = Path('yolo_combined_dataset_symlinks_20251008_162350/test/images')
    
    if not test_images_dir.exists():
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # Get all test images (skip HEIC files as YOLO can't read them)
    test_images = [f for f in test_images_dir.glob('*') 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()]
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📁 Found {len(test_images)} test images")
    
    # Randomly sample images
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    # Create output directory
    output_dir = Path('corner_detection_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 Visualizing {len(sample_images)} sample detections...\n")
    
    for i, image_path in enumerate(sample_images, 1):
        try:
            # Run detection
            results = model(str(image_path), verbose=False)
            
            # Load original image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ Could not load {image_path.name}")
                continue
            
            # Draw detection results
            if results and len(results) > 0:
                result = results[0]
                
                # Get bounding box
                if result.boxes and len(result.boxes) > 0:
                    box = result.boxes[0]
                    
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Calculate corners from bounding box
                    corners = [
                        (x1, y1),  # Top-left
                        (x2, y1),  # Top-right
                        (x2, y2),  # Bottom-right
                        (x1, y2)   # Bottom-left
                    ]
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw corners
                    for j, (cx, cy) in enumerate(corners):
                        # Draw corner circle
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
                        # Draw corner number
                        cv2.putText(img, str(j+1), (cx-10, cy-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    
                    # Draw confidence
                    label = f"Chessboard: {confidence:.1%}"
                    cv2.putText(img, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Save visualization
                    output_path = output_dir / f"detection_{i}_{image_path.stem}.jpg"
                    cv2.imwrite(str(output_path), img)
                    
                    print(f"✅ {i}/{len(sample_images)}: {image_path.name}")
                    print(f"   Confidence: {confidence:.1%}")
                    print(f"   Corners: TL({x1},{y1}), TR({x2},{y1}), BR({x2},{y2}), BL({x1},{y2})")
                    print(f"   Saved: {output_path}")
                    print()
                else:
                    print(f"⚠️  {i}/{len(sample_images)}: {image_path.name} - No chessboard detected")
            else:
                print(f"⚠️  {i}/{len(sample_images)}: {image_path.name} - No results")
                
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
    
    print("=" * 60)
    print(f"✅ Visualizations saved to: {output_dir.absolute()}")
    print(f"📊 Total visualizations: {len(list(output_dir.glob('*.jpg')))}")
    print()
    print("Open the folder to view the corner detections!")

if __name__ == '__main__':
    # New improved model
    new_model = 'yolo_training_runs/improved_corner_detection_20251008_162530/weights/best.pt'
    
    print("=" * 60)
    print("CORNER DETECTION VISUALIZATION")
    print("=" * 60)
    print()
    print(f"Model: {new_model}")
    print()
    
    visualize_corner_detections(new_model, num_samples=10)


