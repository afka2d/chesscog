#!/usr/bin/env python3
"""
Create visual comparisons of OptimizedCornerService vs ground truth corners.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from optimized_corner_service import OptimizedCornerService
from corner_detection_service import CornerDetectionService

def create_optimized_corner_comparison(image_path, annotation_path, output_path):
    """Create visual comparison of original vs optimized vs ground truth corners"""
    
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
    
    # Get corner predictions
    original_service = CornerDetectionService()
    optimized_service = OptimizedCornerService()
    
    original_corners = original_service.detect_corners(image_path)
    optimized_corners = optimized_service.detect_corners(image_path)
    
    if not original_corners or not optimized_corners:
        print(f"❌ Could not detect corners")
        return False
    
    original_corners = np.array(original_corners)
    optimized_corners = np.array(optimized_corners)
    
    # Create three-way comparison
    h, w = image.shape[:2]
    
    # Create canvas for three images side by side
    canvas = np.zeros((h, w * 3 + 100, 3), dtype=np.uint8)
    
    # Ground truth image (left)
    gt_image = image.copy()
    
    # Draw ground truth corners (bright green)
    for i, corner in enumerate(gt_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(gt_image, (x, y), 30, (0, 255, 0), -1)  # Bright green fill
        cv2.circle(gt_image, (x, y), 35, (255, 255, 255), 4)  # White outline
        cv2.putText(gt_image, f'GT{i}', (x-20, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw ground truth board outline
    gt_corners_int = gt_corners.astype(np.int32)
    cv2.polylines(gt_image, [gt_corners_int], True, (0, 255, 0), 5)
    
    # Add title
    cv2.putText(gt_image, "GROUND TRUTH", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # Original model image (middle)
    orig_image = image.copy()
    
    # Draw original corners (red)
    for i, corner in enumerate(original_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(orig_image, (x, y), 25, (0, 0, 255), -1)  # Red fill
        cv2.circle(orig_image, (x, y), 30, (255, 255, 255), 3)  # White outline
        cv2.putText(orig_image, f'O{i}', (x-15, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw original board outline
    orig_corners_int = original_corners.astype(np.int32)
    cv2.polylines(orig_image, [orig_corners_int], True, (0, 0, 255), 4)
    
    # Add title
    cv2.putText(orig_image, "ORIGINAL MODEL", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Optimized model image (right)
    opt_image = image.copy()
    
    # Draw optimized corners (blue)
    for i, corner in enumerate(optimized_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(opt_image, (x, y), 25, (255, 0, 0), -1)  # Blue fill
        cv2.circle(opt_image, (x, y), 30, (255, 255, 255), 3)  # White outline
        cv2.putText(opt_image, f'OP{i}', (x-20, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw optimized board outline
    opt_corners_int = optimized_corners.astype(np.int32)
    cv2.polylines(opt_image, [opt_corners_int], True, (255, 0, 0), 4)
    
    # Add title
    cv2.putText(opt_image, "OPTIMIZED MODEL", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    
    # Place images on canvas
    canvas[:, :w] = gt_image
    canvas[:, w+50:2*w+50] = orig_image
    canvas[:, 2*w+100:] = opt_image
    
    # Add separator lines
    cv2.line(canvas, (w + 25, 0), (w + 25, h), (255, 255, 255), 3)
    cv2.line(canvas, (2*w + 75, 0), (2*w + 75, h), (255, 255, 255), 3)
    
    # Calculate and display accuracy metrics
    orig_errors = np.sqrt(np.sum((gt_corners - original_corners) ** 2, axis=1))
    opt_errors = np.sqrt(np.sum((gt_corners - optimized_corners) ** 2, axis=1))
    
    orig_avg_error = np.mean(orig_errors)
    opt_avg_error = np.mean(opt_errors)
    improvement = orig_avg_error - opt_avg_error
    improvement_pct = (improvement / orig_avg_error) * 100
    
    # Add accuracy info at the bottom
    y_base = h - 120
    cv2.putText(canvas, f"ACCURACY COMPARISON", (w//2, y_base), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(canvas, f"Original: {orig_avg_error:.1f}px", (w//4, y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(canvas, f"Optimized: {opt_avg_error:.1f}px", (2*w + w//4, y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    
    improvement_color = (0, 255, 0) if improvement > 0 else (0, 255, 255)
    cv2.putText(canvas, f"Improvement: {improvement:+.1f}px ({improvement_pct:+.1f}%)", (w//2, y_base + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, improvement_color, 3)
    
    # Add individual corner errors
    for i in range(4):
        corner_improvement = orig_errors[i] - opt_errors[i]
        corner_text = f"C{i}: {orig_errors[i]:.1f}→{opt_errors[i]:.1f} ({corner_improvement:+.1f})"
        cv2.putText(canvas, corner_text, (50 + i * 400, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save comparison
    cv2.imwrite(output_path, canvas)
    
    return {
        'gt_corners': gt_corners.tolist(),
        'original_corners': original_corners.tolist(),
        'optimized_corners': optimized_corners.tolist(),
        'original_errors': orig_errors.tolist(),
        'optimized_errors': opt_errors.tolist(),
        'original_avg_error': orig_avg_error,
        'optimized_avg_error': opt_avg_error,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'output_path': output_path
    }

def create_overlay_comparison(image_path, annotation_path, output_path):
    """Create overlay showing all three corner sets on the same image"""
    
    # Load image
    image = cv2.imread(image_path)
    
    # Load ground truth
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Get predictions
    original_service = CornerDetectionService()
    optimized_service = OptimizedCornerService()
    
    original_corners = np.array(original_service.detect_corners(image_path))
    optimized_corners = np.array(optimized_service.detect_corners(image_path))
    
    # Create overlay
    overlay_image = image.copy()
    
    # Draw ground truth corners (large bright green circles)
    for i, corner in enumerate(gt_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 40, (0, 255, 0), -1)
        cv2.circle(overlay_image, (x, y), 45, (255, 255, 255), 4)
        cv2.putText(overlay_image, f'GT{i}', (x-25, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    
    # Draw original corners (medium red circles)
    for i, corner in enumerate(original_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 30, (0, 0, 255), -1)
        cv2.circle(overlay_image, (x, y), 35, (255, 255, 255), 3)
        cv2.putText(overlay_image, f'OR{i}', (x+50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw optimized corners (small blue circles)
    for i, corner in enumerate(optimized_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(overlay_image, (x, y), 20, (255, 0, 0), -1)
        cv2.circle(overlay_image, (x, y), 25, (255, 255, 255), 2)
        cv2.putText(overlay_image, f'OP{i}', (x-50, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw board outlines
    gt_corners_int = gt_corners.astype(np.int32)
    orig_corners_int = original_corners.astype(np.int32)
    opt_corners_int = optimized_corners.astype(np.int32)
    
    cv2.polylines(overlay_image, [gt_corners_int], True, (0, 255, 0), 6)    # Thick green for GT
    cv2.polylines(overlay_image, [orig_corners_int], True, (0, 0, 255), 4)  # Medium red for original
    cv2.polylines(overlay_image, [opt_corners_int], True, (255, 0, 0), 3)   # Thin blue for optimized
    
    # Add legend
    legend_y = 150
    cv2.putText(overlay_image, "LEGEND:", (50, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(overlay_image, "Large Green = Ground Truth", (50, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(overlay_image, "Medium Red = Original Model", (50, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(overlay_image, "Small Blue = Optimized Model", (50, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Calculate and display accuracy
    orig_errors = np.sqrt(np.sum((gt_corners - original_corners) ** 2, axis=1))
    opt_errors = np.sqrt(np.sum((gt_corners - optimized_corners) ** 2, axis=1))
    
    orig_avg = np.mean(orig_errors)
    opt_avg = np.mean(opt_errors)
    improvement = orig_avg - opt_avg
    improvement_pct = (improvement / orig_avg) * 100
    
    # Add accuracy text
    h, w = image.shape[:2]
    cv2.putText(overlay_image, f"Original: {orig_avg:.1f}px | Optimized: {opt_avg:.1f}px", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    improvement_color = (0, 255, 0) if improvement > 0 else (0, 255, 255)
    cv2.putText(overlay_image, f"Improvement: {improvement:+.1f}px ({improvement_pct:+.1f}%)", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, improvement_color, 3)
    
    # Save overlay
    cv2.imwrite(output_path, overlay_image)
    
    return {
        'original_avg_error': orig_avg,
        'optimized_avg_error': opt_avg,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

def create_detailed_corner_analysis(image_path, annotation_path, output_path):
    """Create detailed per-corner analysis visualization"""
    
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Load ground truth
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Get predictions
    original_service = CornerDetectionService()
    optimized_service = OptimizedCornerService()
    
    original_corners = np.array(original_service.detect_corners(image_path))
    optimized_corners = np.array(optimized_service.detect_corners(image_path))
    
    # Create detailed analysis image
    analysis_image = image.copy()
    
    # Draw connections between corresponding corners
    corner_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 255)]  # Cyan, Magenta, Yellow, Light cyan
    
    for i, (gt, orig, opt, color) in enumerate(zip(gt_corners, original_corners, optimized_corners, corner_colors)):
        gt_point = tuple(gt.astype(int))
        orig_point = tuple(orig.astype(int))
        opt_point = tuple(opt.astype(int))
        
        # Draw connection lines
        cv2.line(analysis_image, gt_point, orig_point, (0, 0, 255), 2)  # GT to Original (red)
        cv2.line(analysis_image, gt_point, opt_point, (255, 0, 0), 2)   # GT to Optimized (blue)
        
        # Draw corner points
        cv2.circle(analysis_image, gt_point, 25, (0, 255, 0), -1)      # GT (green)
        cv2.circle(analysis_image, orig_point, 20, (0, 0, 255), -1)    # Original (red)
        cv2.circle(analysis_image, opt_point, 15, (255, 0, 0), -1)     # Optimized (blue)
        
        # Add corner labels
        cv2.putText(analysis_image, f'C{i}', gt_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate and display errors
        orig_error = np.linalg.norm(gt - orig)
        opt_error = np.linalg.norm(gt - opt)
        
        # Error text position
        text_x = gt_point[0] + 60
        text_y = gt_point[1]
        
        cv2.putText(analysis_image, f'Orig: {orig_error:.1f}px', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(analysis_image, f'Opt: {opt_error:.1f}px', (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        improvement = orig_error - opt_error
        improvement_color = (0, 255, 0) if improvement > 0 else (0, 255, 255)
        cv2.putText(analysis_image, f'Δ: {improvement:+.1f}px', (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, improvement_color, 2)
    
    # Add title and legend
    cv2.putText(analysis_image, "DETAILED CORNER ANALYSIS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    legend_y = h - 150
    cv2.putText(analysis_image, "Green=Ground Truth, Red=Original, Blue=Optimized", (50, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(analysis_image, "Lines show error vectors from Ground Truth", (50, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save analysis
    cv2.imwrite(output_path, analysis_image)

def main():
    """Create comprehensive visual comparisons"""
    print("🎨 CREATING OPTIMIZED CORNER DETECTION VISUALIZATIONS")
    print("=" * 60)
    print("Comparing Original vs Optimized vs Ground Truth corners")
    print()
    
    # Test cases
    test_cases = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json',
            'output_comparison': 'optimized_comparison_IMG_4779.jpg',
            'output_overlay': 'optimized_overlay_IMG_4779.jpg',
            'output_analysis': 'optimized_analysis_IMG_4779.jpg'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json',
            'output_comparison': 'optimized_comparison_IMG_4785.jpg',
            'output_overlay': 'optimized_overlay_IMG_4785.jpg',
            'output_analysis': 'optimized_analysis_IMG_4785.jpg'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json',
            'output_comparison': 'optimized_comparison_IMG_4763.jpg',
            'output_overlay': 'optimized_overlay_IMG_4763.jpg',
            'output_analysis': 'optimized_analysis_IMG_4763.jpg'
        }
    ]
    
    successful_comparisons = 0
    total_original_error = 0
    total_optimized_error = 0
    
    for i, test_case in enumerate(test_cases):
        if not Path(test_case['image']).exists() or not Path(test_case['annotation']).exists():
            print(f"⚠️  Skipping {test_case['image']} - files not found")
            continue
        
        print(f"📸 Creating visualizations {i+1}: {Path(test_case['image']).name}")
        
        # Create three-way comparison
        result = create_optimized_corner_comparison(
            test_case['image'],
            test_case['annotation'],
            test_case['output_comparison']
        )
        
        if result:
            print(f"   ✅ Three-way comparison: {test_case['output_comparison']}")
            print(f"   📊 Original: {result['original_avg_error']:.1f}px | Optimized: {result['optimized_avg_error']:.1f}px")
            print(f"   📈 Improvement: {result['improvement']:+.1f}px ({result['improvement_pct']:+.1f}%)")
            
            # Create overlay comparison
            create_overlay_comparison(
                test_case['image'],
                test_case['annotation'],
                test_case['output_overlay']
            )
            print(f"   ✅ Overlay comparison: {test_case['output_overlay']}")
            
            # Create detailed analysis
            create_detailed_corner_analysis(
                test_case['image'],
                test_case['annotation'],
                test_case['output_analysis']
            )
            print(f"   ✅ Detailed analysis: {test_case['output_analysis']}")
            
            successful_comparisons += 1
            total_original_error += result['original_avg_error']
            total_optimized_error += result['optimized_avg_error']
        else:
            print(f"   ❌ Failed to create comparison")
        
        print()
    
    # Overall summary
    if successful_comparisons > 0:
        avg_original = total_original_error / successful_comparisons
        avg_optimized = total_optimized_error / successful_comparisons
        overall_improvement = avg_original - avg_optimized
        overall_improvement_pct = (overall_improvement / avg_original) * 100
        
        print(f"📊 OVERALL OPTIMIZED CORNER DETECTION SUMMARY:")
        print(f"   Successful visualizations: {successful_comparisons}")
        print(f"   Average original error: {avg_original:.1f} pixels")
        print(f"   Average optimized error: {avg_optimized:.1f} pixels")
        print(f"   Overall improvement: {overall_improvement:+.1f} pixels ({overall_improvement_pct:+.1f}%)")
        
        if overall_improvement > 5:
            print("   🎯 SIGNIFICANT IMPROVEMENT!")
        elif overall_improvement > 2:
            print("   ✅ GOOD IMPROVEMENT")
        elif overall_improvement > 0:
            print("   ✅ MARGINAL IMPROVEMENT")
        else:
            print("   ⚠️  NO IMPROVEMENT")
        
        print(f"\n🎨 VISUALIZATION FILES CREATED:")
        for test_case in test_cases:
            for output_key in ['output_comparison', 'output_overlay', 'output_analysis']:
                output_file = test_case[output_key]
                if Path(output_file).exists():
                    print(f"   📸 {output_file}")
        
        print(f"\n💡 HOW TO INTERPRET:")
        print("   Three-way comparison: Side by side comparison")
        print("   Overlay: All corners on same image")
        print("   Analysis: Detailed per-corner error vectors")
        print("   Green = Ground Truth, Red = Original, Blue = Optimized")
        
        return True
    else:
        print("❌ No successful visualizations created")
        return False

if __name__ == "__main__":
    main()
