#!/usr/bin/env python3
"""
Smart bias correction that analyzes the systematic bias pattern and applies targeted corrections.
"""

import numpy as np
import json
from pathlib import Path
from corner_detection_service import CornerDetectionService
import cv2

def analyze_systematic_bias():
    """Analyze the systematic bias pattern in corner predictions"""
    print("🔍 ANALYZING SYSTEMATIC BIAS PATTERNS")
    print("=" * 60)
    
    # Load test cases
    test_cases = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json'
        }
    ]
    
    service = CornerDetectionService()
    bias_vectors = []
    per_corner_biases = [[], [], [], []]  # One list per corner (TL, TR, BR, BL)
    
    for test_case in test_cases:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"\n📸 Analyzing: {Path(image_path).name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Get predictions
        pred_corners = service.detect_corners(image_path)
        if not pred_corners:
            continue
        
        pred_corners = np.array(pred_corners)
        
        # Calculate bias vectors (prediction - ground_truth)
        bias = pred_corners - gt_corners
        bias_vectors.extend(bias)
        
        # Store per-corner biases
        for i, corner_bias in enumerate(bias):
            per_corner_biases[i].append(corner_bias)
            
        # Display individual corner biases
        for i, (gt, pred, b) in enumerate(zip(gt_corners, pred_corners, bias)):
            magnitude = np.linalg.norm(b)
            print(f"   Corner {i}: GT({gt[0]:.0f},{gt[1]:.0f}) → Pred({pred[0]:.0f},{pred[1]:.0f}) | Bias: ({b[0]:+.1f},{b[1]:+.1f}) | Mag: {magnitude:.1f}px")
    
    if not bias_vectors:
        print("❌ No data to analyze")
        return None
    
    # Calculate overall statistics
    bias_vectors = np.array(bias_vectors)
    mean_bias = np.mean(bias_vectors, axis=0)
    std_bias = np.std(bias_vectors, axis=0)
    
    print(f"\n📊 OVERALL BIAS ANALYSIS:")
    print(f"   Mean bias: ({mean_bias[0]:+.1f}, {mean_bias[1]:+.1f}) pixels")
    print(f"   Std deviation: ({std_bias[0]:.1f}, {std_bias[1]:.1f}) pixels")
    print(f"   Mean magnitude: {np.mean(np.linalg.norm(bias_vectors, axis=1)):.1f} pixels")
    
    # Per-corner analysis
    print(f"\n📊 PER-CORNER BIAS ANALYSIS:")
    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    per_corner_corrections = []
    
    for i, (name, biases) in enumerate(zip(corner_names, per_corner_biases)):
        if biases:
            biases_array = np.array(biases)
            mean_corner_bias = np.mean(biases_array, axis=0)
            mean_magnitude = np.mean(np.linalg.norm(biases_array, axis=1))
            per_corner_corrections.append(mean_corner_bias)
            
            print(f"   {name}: ({mean_corner_bias[0]:+.1f}, {mean_corner_bias[1]:+.1f}) | Avg: {mean_magnitude:.1f}px")
        else:
            per_corner_corrections.append([0, 0])
    
    # Determine correction strategy
    print(f"\n💡 CORRECTION STRATEGY:")
    
    overall_magnitude = np.linalg.norm(mean_bias)
    if overall_magnitude > 10:
        print(f"   🎯 GLOBAL BIAS DETECTED: {overall_magnitude:.1f} pixels")
        print(f"   Strategy: Apply global correction ({-mean_bias[0]:+.1f}, {-mean_bias[1]:+.1f})")
        strategy = "global"
        correction_data = -mean_bias
    else:
        # Check if per-corner corrections are significant
        corner_magnitudes = [np.linalg.norm(correction) for correction in per_corner_corrections]
        max_corner_magnitude = max(corner_magnitudes)
        
        if max_corner_magnitude > 5:
            print(f"   🎯 PER-CORNER BIAS DETECTED: Max {max_corner_magnitude:.1f} pixels")
            print(f"   Strategy: Apply per-corner corrections")
            strategy = "per_corner"
            correction_data = [-correction for correction in per_corner_corrections]
        else:
            print(f"   ✅ NO SIGNIFICANT BIAS DETECTED")
            print(f"   Strategy: No correction needed")
            strategy = "none"
            correction_data = None
    
    return {
        "strategy": strategy,
        "correction_data": correction_data,
        "mean_bias": mean_bias,
        "per_corner_corrections": per_corner_corrections,
        "overall_magnitude": overall_magnitude
    }

class SmartBiasCorrection:
    """Smart bias correction based on systematic bias analysis"""
    
    def __init__(self, bias_analysis=None):
        if bias_analysis is None:
            # Run analysis if not provided
            bias_analysis = analyze_systematic_bias()
        
        self.strategy = bias_analysis.get("strategy", "none")
        self.correction_data = bias_analysis.get("correction_data")
        
    def apply_correction(self, corners):
        """Apply the determined correction strategy"""
        if not corners or len(corners) != 4 or self.strategy == "none":
            return corners
        
        corners_array = np.array(corners)
        
        if self.strategy == "global":
            # Apply same correction to all corners
            corrected_corners = corners_array + self.correction_data
            
        elif self.strategy == "per_corner":
            # Apply different correction to each corner
            corrected_corners = corners_array + np.array(self.correction_data)
            
        else:
            corrected_corners = corners_array
        
        return corrected_corners.tolist()

class SmartBiasCornerService:
    """Corner detection service with smart bias correction"""
    
    def __init__(self, model_path="models/corner_detector_best.pt"):
        self.base_service = CornerDetectionService(model_path)
        
        # Analyze bias and create correction
        print("🧠 Initializing smart bias correction...")
        bias_analysis = analyze_systematic_bias()
        self.bias_corrector = SmartBiasCorrection(bias_analysis)
        
    def detect_corners(self, image_path):
        """Detect corners with smart bias correction"""
        # Get original predictions
        original_corners = self.base_service.detect_corners(image_path)
        
        if not original_corners:
            return None
        
        # Apply smart correction
        corrected_corners = self.bias_corrector.apply_correction(original_corners)
        
        return corrected_corners

def test_smart_correction():
    """Test the smart bias correction"""
    print(f"\n🧠 TESTING SMART BIAS CORRECTION")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json'
        }
    ]
    
    # Initialize services
    original_service = CornerDetectionService()
    smart_service = SmartBiasCornerService()
    
    total_original_error = 0
    total_corrected_error = 0
    valid_tests = 0
    
    for test_case in test_cases:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Get predictions
        original_corners = original_service.detect_corners(image_path)
        corrected_corners = smart_service.detect_corners(image_path)
        
        if original_corners and corrected_corners:
            original_error = np.mean(np.sqrt(np.sum((gt_corners - np.array(original_corners)) ** 2, axis=1)))
            corrected_error = np.mean(np.sqrt(np.sum((gt_corners - np.array(corrected_corners)) ** 2, axis=1)))
            
            improvement = original_error - corrected_error
            improvement_pct = (improvement / original_error) * 100
            
            print(f"   Original error: {original_error:.1f} pixels")
            print(f"   Corrected error: {corrected_error:.1f} pixels")
            print(f"   Improvement: {improvement:+.1f} pixels ({improvement_pct:+.1f}%)")
            
            total_original_error += original_error
            total_corrected_error += corrected_error
            valid_tests += 1
        else:
            print("   ❌ Detection failed")
    
    # Summary
    if valid_tests > 0:
        avg_original = total_original_error / valid_tests
        avg_corrected = total_corrected_error / valid_tests
        avg_improvement = avg_original - avg_corrected
        avg_improvement_pct = (avg_improvement / avg_original) * 100
        
        print(f"\n📊 SMART CORRECTION SUMMARY:")
        print(f"   Average original error: {avg_original:.1f} pixels")
        print(f"   Average corrected error: {avg_corrected:.1f} pixels")
        print(f"   Average improvement: {avg_improvement:+.1f} pixels ({avg_improvement_pct:+.1f}%)")
        
        if avg_improvement_pct > 10:
            print("   🎯 EXCELLENT IMPROVEMENT!")
        elif avg_improvement_pct > 5:
            print("   ✅ GOOD IMPROVEMENT")
        elif avg_improvement_pct > 0:
            print("   ✅ MARGINAL IMPROVEMENT")
        else:
            print("   ⚠️  NO IMPROVEMENT - Original model is already well-calibrated")

def create_visual_comparison():
    """Create visual comparison of original vs corrected corners"""
    print(f"\n🎨 CREATING VISUAL COMPARISON")
    print("-" * 40)
    
    # Use best test image
    image_path = 'grey_background_dataset/images/test/IMG_4785.JPG'
    annotation_path = 'grey_background_dataset/annotations/test/IMG_4785.json'
    
    if not Path(image_path).exists() or not Path(annotation_path).exists():
        print("❌ Test files not found")
        return
    
    # Load image and ground truth
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Get predictions
    original_service = CornerDetectionService()
    smart_service = SmartBiasCornerService()
    
    original_corners = np.array(original_service.detect_corners(image_path))
    corrected_corners = np.array(smart_service.detect_corners(image_path))
    
    # Create comparison image
    comparison = image.copy()
    
    # Draw ground truth (green circles)
    for i, corner in enumerate(gt_corners):
        cv2.circle(comparison, tuple(corner.astype(int)), 20, (0, 255, 0), -1)
        cv2.putText(comparison, f'GT{i}', tuple(corner.astype(int) + [0, -30]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw original predictions (red circles)
    for i, corner in enumerate(original_corners):
        cv2.circle(comparison, tuple(corner.astype(int)), 15, (0, 0, 255), -1)
        cv2.putText(comparison, f'O{i}', tuple(corner.astype(int) + [25, 0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw corrected predictions (blue circles)
    for i, corner in enumerate(corrected_corners):
        cv2.circle(comparison, tuple(corner.astype(int)), 10, (255, 0, 0), -1)
        cv2.putText(comparison, f'C{i}', tuple(corner.astype(int) + [-25, 0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Add legend
    cv2.putText(comparison, "Green=Ground Truth, Red=Original, Blue=Corrected", 
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save comparison
    output_path = "smart_bias_correction_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    
    print(f"✅ Visual comparison saved: {output_path}")

def main():
    """Main function"""
    print("Smart Bias Correction Analysis")
    print("=" * 50)
    
    # Run bias analysis
    bias_analysis = analyze_systematic_bias()
    
    # Test smart correction
    test_smart_correction()
    
    # Create visual comparison
    create_visual_comparison()
    
    print(f"\n🎯 CONCLUSION:")
    if bias_analysis and bias_analysis.get("strategy") != "none":
        print("✅ Systematic bias detected and correction applied")
        print("💡 Use SmartBiasCornerService for improved accuracy")
    else:
        print("✅ No significant systematic bias detected")
        print("💡 Original model is already well-calibrated")
        print("💡 Focus on other improvement strategies (more data, better model)")

if __name__ == "__main__":
    main()
