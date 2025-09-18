#!/usr/bin/env python3
"""
Create overlay comparison showing both corner sets on the same image.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from corner_detection_service import CornerDetectionService

def create_overlay_for_image(image_path, annotation_path, output_path):
    """Create overlay comparison for a single image"""
    
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Load ground truth
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Detect corners
    service = CornerDetectionService()
    pred_corners = np.array(service.detect_corners(image_path))
    
    # Create overlay
    overlay_image = image.copy()
    
    # Draw ground truth corners (larger, bright colors)
    gt_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 255, 255)]  # Bright colors
    gt_labels = ['GT-TL', 'GT-TR', 'GT-BR', 'GT-BL']
    
    for i, (corner, color, label) in enumerate(zip(gt_corners, gt_colors, gt_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 40, color, -1)
        cv2.circle(overlay_image, (x, y), 45, (255, 255, 255), 4)
        cv2.putText(overlay_image, label, (x-50, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    # Draw predicted corners (smaller, different colors)
    pred_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 128, 0)]
    pred_labels = ['AI-TL', 'AI-TR', 'AI-BR', 'AI-BL']
    
    for i, (corner, color, label) in enumerate(zip(pred_corners, pred_colors, pred_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 25, color, -1)
        cv2.circle(overlay_image, (x, y), 30, (0, 0, 0), 3)
        cv2.putText(overlay_image, label, (x-30, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw both board outlines
    gt_corners_int = gt_corners.astype(np.int32)
    pred_corners_int = pred_corners.astype(np.int32)
    
    cv2.polylines(overlay_image, [gt_corners_int], True, (0, 255, 255), 6)  # Thick cyan for GT
    cv2.polylines(overlay_image, [pred_corners_int], True, (0, 0, 255), 4)  # Red for predicted
    
    # Add title and legend
    cv2.putText(overlay_image, "CORNER DETECTION COMPARISON", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    legend_y = 120
    cv2.putText(overlay_image, "Large circles + Cyan = Ground Truth", (50, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(overlay_image, "Small circles + Red = AI Detected", (50, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Calculate and display accuracy
    errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
    avg_error = np.mean(errors)
    
    cv2.putText(overlay_image, f"Average Error: {avg_error:.1f} pixels", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Add individual corner errors
    for i, error in enumerate(errors):
        error_text = f"Corner {i+1}: {error:.1f}px"
        cv2.putText(overlay_image, error_text, (50 + i * 400, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save overlay
    cv2.imwrite(output_path, overlay_image)
    
    return avg_error

def main():
    """Main function"""
    print("Creating Overlay Corner Comparisons")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json',
            'output': 'overlay_IMG_4779.jpg'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json',
            'output': 'overlay_IMG_4785.jpg'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json',
            'output': 'overlay_IMG_4763.jpg'
        }
    ]
    
    successful_overlays = 0
    total_error = 0
    
    for i, test_case in enumerate(test_cases):
        if not Path(test_case['image']).exists() or not Path(test_case['annotation']).exists():
            continue
        
        print(f"\n📸 Creating overlay {i+1}: {Path(test_case['image']).name}")
        
        try:
            avg_error = create_overlay_for_image(
                test_case['image'],
                test_case['annotation'],
                test_case['output']
            )
            
            print(f"   ✅ Overlay saved: {test_case['output']}")
            print(f"   📊 Average error: {avg_error:.1f} pixels")
            
            successful_overlays += 1
            total_error += avg_error
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if successful_overlays > 0:
        overall_avg = total_error / successful_overlays
        
        print(f"\n📊 OVERLAY SUMMARY:")
        print(f"   Successful overlays: {successful_overlays}")
        print(f"   Overall average error: {overall_avg:.1f} pixels")
        
        print(f"\n🎨 OVERLAY FILES CREATED:")
        for test_case in test_cases:
            if Path(test_case['output']).exists():
                print(f"   📸 {test_case['output']}")
        
        print(f"\n💡 HOW TO VIEW THE OVERLAYS:")
        print("   - Large bright circles = Ground Truth corners")
        print("   - Small darker circles = AI Detected corners")
        print("   - Cyan outline = Ground Truth board")
        print("   - Red outline = AI Detected board")
        print("   - Closer the circles = better accuracy")

if __name__ == "__main__":
    main()
