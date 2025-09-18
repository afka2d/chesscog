#!/usr/bin/env python3
"""
Visual comparison tool to show ground truth vs automatically detected corners.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from corner_detection_service import CornerDetectionService

def create_corner_comparison(image_path, annotation_path, output_path):
    """Create side-by-side comparison of ground truth vs detected corners"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    # Load ground truth
    try:
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
    except Exception as e:
        print(f"❌ Could not load ground truth: {e}")
        return False
    
    # Detect corners automatically
    service = CornerDetectionService()
    pred_corners = service.detect_corners(image_path)
    
    if pred_corners is None:
        print(f"❌ Could not detect corners")
        return False
    
    pred_corners = np.array(pred_corners)
    
    # Create side-by-side comparison
    h, w = image.shape[:2]
    
    # Create canvas for side-by-side images
    canvas = np.zeros((h, w * 2 + 50, 3), dtype=np.uint8)
    
    # Ground truth image (left side)
    gt_image = image.copy()
    
    # Draw ground truth corners
    gt_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 255)]  # Cyan, Magenta, Yellow, Light Purple
    gt_labels = ['GT-TL', 'GT-TR', 'GT-BR', 'GT-BL']
    
    for i, (corner, color, label) in enumerate(zip(gt_corners, gt_colors, gt_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(gt_image, (x, y), 25, color, -1)
        cv2.circle(gt_image, (x, y), 30, (255, 255, 255), 3)  # White outline
        cv2.putText(gt_image, label, (x-30, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw ground truth board outline
    gt_corners_int = gt_corners.astype(np.int32)
    cv2.polylines(gt_image, [gt_corners_int], True, (0, 255, 255), 4)
    
    # Add title
    cv2.putText(gt_image, "GROUND TRUTH CORNERS", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    # Predicted image (right side)
    pred_image = image.copy()
    
    # Draw predicted corners
    pred_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
    pred_labels = ['AI-TL', 'AI-TR', 'AI-BR', 'AI-BL']
    
    for i, (corner, color, label) in enumerate(zip(pred_corners, pred_colors, pred_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(pred_image, (x, y), 25, color, -1)
        cv2.circle(pred_image, (x, y), 30, (255, 255, 255), 3)  # White outline
        cv2.putText(pred_image, label, (x-30, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw predicted board outline
    pred_corners_int = pred_corners.astype(np.int32)
    cv2.polylines(pred_image, [pred_corners_int], True, (0, 0, 255), 4)
    
    # Add title
    cv2.putText(pred_image, "AI DETECTED CORNERS", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Place images on canvas
    canvas[:, :w] = gt_image
    canvas[:, w+50:] = pred_image
    
    # Add separator line
    cv2.line(canvas, (w + 25, 0), (w + 25, h), (255, 255, 255), 2)
    
    # Calculate and display accuracy
    errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
    avg_error = np.mean(errors)
    
    # Add accuracy info at the bottom
    accuracy_text = f"Average Error: {avg_error:.1f} pixels"
    cv2.putText(canvas, accuracy_text, (w//2 - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add individual corner errors
    for i, error in enumerate(errors):
        error_text = f"Corner {i+1}: {error:.1f}px"
        cv2.putText(canvas, error_text, (50 + i * 300, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save comparison
    cv2.imwrite(output_path, canvas)
    
    return {
        'gt_corners': gt_corners.tolist(),
        'pred_corners': pred_corners.tolist(),
        'errors': errors.tolist(),
        'avg_error': avg_error,
        'output_path': output_path
    }

def create_multiple_comparisons():
    """Create comparison visualizations for multiple images"""
    print("🎨 CREATING VISUAL CORNER COMPARISONS")
    print("=" * 60)
    print("This will show ground truth vs AI detected corners side by side")
    print()
    
    # Find test images with annotations
    test_cases = [
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json',
            'output': 'comparison_IMG_4785.jpg'
        },
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json',
            'output': 'comparison_IMG_4779.jpg'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json',
            'output': 'comparison_IMG_4763.jpg'
        }
    ]
    
    successful_comparisons = 0
    total_error = 0
    
    for i, test_case in enumerate(test_cases):
        if not Path(test_case['image']).exists() or not Path(test_case['annotation']).exists():
            print(f"⚠️  Skipping {test_case['image']} - files not found")
            continue
        
        print(f"📸 Creating comparison {i+1}: {Path(test_case['image']).name}")
        
        result = create_corner_comparison(
            test_case['image'],
            test_case['annotation'],
            test_case['output']
        )
        
        if result:
            print(f"   ✅ Comparison saved: {test_case['output']}")
            print(f"   📊 Average error: {result['avg_error']:.1f} pixels")
            print(f"   🎯 Per-corner errors: {[f'{e:.1f}' for e in result['errors']]} pixels")
            
            successful_comparisons += 1
            total_error += result['avg_error']
            
            # Show corner coordinates
            print(f"   📍 Ground Truth: {result['gt_corners']}")
            print(f"   🤖 AI Detected: {result['pred_corners']}")
            
        else:
            print(f"   ❌ Failed to create comparison")
        
        print()
    
    # Summary
    if successful_comparisons > 0:
        overall_avg_error = total_error / successful_comparisons
        
        print(f"📊 OVERALL COMPARISON SUMMARY:")
        print(f"   Successful comparisons: {successful_comparisons}")
        print(f"   Overall average error: {overall_avg_error:.1f} pixels")
        
        if overall_avg_error < 50:
            print("   ✅ EXCELLENT: Very accurate corner detection")
        elif overall_avg_error < 100:
            print("   ✅ GOOD: Acceptable corner detection accuracy")
        elif overall_avg_error < 200:
            print("   ⚠️  FAIR: Corner detection needs improvement")
        else:
            print("   ❌ POOR: Significant improvement needed")
        
        print(f"\n🎨 VISUALIZATION FILES CREATED:")
        for test_case in test_cases:
            if Path(test_case['output']).exists():
                print(f"   📸 {test_case['output']}")
        
        print(f"\n💡 HOW TO VIEW:")
        print("   Open the comparison images to see:")
        print("   - Left side: Ground truth corners (cyan/magenta/yellow)")
        print("   - Right side: AI detected corners (red/green/blue/yellow)")
        print("   - Error measurements at the bottom")
        
        return True
    else:
        print("❌ No successful comparisons created")
        return False

def create_overlay_comparison():
    """Create overlay comparison showing both corner sets on the same image"""
    print(f"\n🎨 CREATING OVERLAY COMPARISON")
    print("-" * 30)
    
    # Use the best performing image
    image_path = 'grey_background_dataset/images/val/IMG_4779.JPG'
    annotation_path = 'grey_background_dataset/annotations/val/IMG_4779.json'
    output_path = 'overlay_comparison_IMG_4779.jpg'
    
    if not Path(image_path).exists() or not Path(annotation_path).exists():
        print(f"⚠️  Files not found for overlay comparison")
        return False
    
    # Load image
    image = cv2.imread(image_path)
    
    # Load ground truth
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Detect corners
    service = CornerDetectionService()
    pred_corners = np.array(service.detect_corners(image_path))
    
    # Create overlay
    overlay_image = image.copy()
    
    # Draw ground truth corners (larger, outlined)
    gt_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 128, 255)]
    gt_labels = ['GT-TL', 'GT-TR', 'GT-BR', 'GT-BL']
    
    for i, (corner, color, label) in enumerate(zip(gt_corners, gt_colors, gt_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 35, color, -1)
        cv2.circle(overlay_image, (x, y), 40, (255, 255, 255), 4)
        cv2.putText(overlay_image, label, (x-40, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    # Draw predicted corners (smaller, inside)
    pred_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 128, 0)]
    pred_labels = ['AI-TL', 'AI-TR', 'AI-BR', 'AI-BL']
    
    for i, (corner, color, label) in enumerate(zip(pred_corners, pred_colors, pred_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 20, color, -1)
        cv2.circle(overlay_image, (x, y), 25, (0, 0, 0), 3)
        cv2.putText(overlay_image, label, (x-25, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw both board outlines
    gt_corners_int = gt_corners.astype(np.int32)
    pred_corners_int = pred_corners.astype(np.int32)
    
    cv2.polylines(overlay_image, [gt_corners_int], True, (0, 255, 255), 5)  # Cyan for GT
    cv2.polylines(overlay_image, [pred_corners_int], True, (0, 0, 255), 3)  # Red for predicted
    
    # Add legend
    legend_y = 150
    cv2.putText(overlay_image, "LEGEND:", (50, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(overlay_image, "Large circles + Cyan outline = Ground Truth", (50, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay_image, "Small circles + Red outline = AI Detected", (50, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Calculate and display accuracy
    errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
    avg_error = np.mean(errors)
    
    cv2.putText(overlay_image, f"Average Error: {avg_error:.1f} pixels", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Save overlay
    cv2.imwrite(output_path, overlay_image)
    
    return {
        'gt_corners': gt_corners.tolist(),
        'pred_corners': pred_corners.tolist(),
        'errors': errors.tolist(),
        'avg_error': avg_error,
        'output_path': output_path
    }

def main():
    """Main function"""
    print("Visual Corner Comparison Tool")
    print("=" * 50)
    print("This creates visual comparisons showing ground truth vs AI detected corners")
    print()
    
    # Create side-by-side comparisons
    success = create_multiple_comparisons()
    
    if success:
        # Create overlay comparison
        overlay_result = create_overlay_comparison()
        
        if overlay_result:
            print(f"✅ Overlay comparison created: {overlay_result['output_path']}")
    
    print(f"\n🎯 VISUAL COMPARISON COMPLETE!")
    print("\nFiles created for your review:")
    
    comparison_files = [
        'comparison_IMG_4785.jpg',
        'comparison_IMG_4779.jpg', 
        'comparison_IMG_4763.jpg',
        'overlay_comparison_IMG_4779.jpg'
    ]
    
    for file_path in comparison_files:
        if Path(file_path).exists():
            print(f"   📸 {file_path}")
    
    print(f"\n💡 HOW TO INTERPRET:")
    print("   - Side-by-side: Left = Ground Truth, Right = AI Detected")
    print("   - Overlay: Large circles = Ground Truth, Small circles = AI")
    print("   - Error measurements show pixel-level accuracy")
    print("   - Closer circles = better accuracy")

if __name__ == "__main__":
    main()
