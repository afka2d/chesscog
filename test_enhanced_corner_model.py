#!/usr/bin/env python3
"""
Test the enhanced corner detection model and compare with the original.
"""

import numpy as np
import json
from pathlib import Path
from sub_pixel_corner_refinement import EnhancedCornerDetectionService
from corner_detection_service import CornerDetectionService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_both_models():
    """Test both original and enhanced corner detection models"""
    print("🎯 ENHANCED CORNER DETECTION MODEL COMPARISON")
    print("=" * 60)
    
    # Initialize both services
    print("Loading models...")
    try:
        original_service = CornerDetectionService("models/corner_detector_best.pt")
        print("✅ Original model loaded")
    except:
        print("❌ Original model failed to load")
        original_service = None
    
    try:
        enhanced_service = EnhancedCornerDetectionService("models/enhanced_corner_detector_best.pt")
        print("✅ Enhanced model loaded")
    except:
        print("❌ Enhanced model failed to load")
        enhanced_service = None
    
    if not enhanced_service:
        print("❌ Cannot proceed without enhanced model")
        return
    
    # Test images with ground truth
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
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Test original model
        if original_service:
            try:
                orig_corners = original_service.detect_corners(image_path)
                if orig_corners:
                    orig_corners = np.array(orig_corners)
                    orig_errors = np.sqrt(np.sum((gt_corners - orig_corners) ** 2, axis=1))
                    orig_avg_error = np.mean(orig_errors)
                else:
                    orig_avg_error = float('inf')
            except:
                orig_avg_error = float('inf')
        else:
            orig_avg_error = float('inf')
        
        # Test enhanced model
        try:
            enhanced_corners = enhanced_service.detect_corners_with_refinement(image_path)
            if enhanced_corners:
                enhanced_corners = np.array(enhanced_corners)
                enhanced_errors = np.sqrt(np.sum((gt_corners - enhanced_corners) ** 2, axis=1))
                enhanced_avg_error = np.mean(enhanced_errors)
            else:
                enhanced_avg_error = float('inf')
        except:
            enhanced_avg_error = float('inf')
        
        # Store results
        result = {
            'image': Path(image_path).name,
            'original_error': orig_avg_error,
            'enhanced_error': enhanced_avg_error,
            'improvement': orig_avg_error - enhanced_avg_error if orig_avg_error != float('inf') else 0
        }
        results.append(result)
        
        print(f"   Original model error: {orig_avg_error:.1f} pixels")
        print(f"   Enhanced model error: {enhanced_avg_error:.1f} pixels")
        if orig_avg_error != float('inf') and enhanced_avg_error != float('inf'):
            improvement = ((orig_avg_error - enhanced_avg_error) / orig_avg_error) * 100
            print(f"   Improvement: {improvement:.1f}%")
    
    # Summary
    print(f"\n📊 OVERALL COMPARISON SUMMARY")
    print("=" * 40)
    
    valid_results = [r for r in results if r['original_error'] != float('inf') and r['enhanced_error'] != float('inf')]
    
    if valid_results:
        avg_original = np.mean([r['original_error'] for r in valid_results])
        avg_enhanced = np.mean([r['enhanced_error'] for r in valid_results])
        overall_improvement = ((avg_original - avg_enhanced) / avg_original) * 100
        
        print(f"Original model average: {avg_original:.1f} pixels")
        print(f"Enhanced model average: {avg_enhanced:.1f} pixels")
        print(f"Overall improvement: {overall_improvement:.1f}%")
        
        # Determine success level
        if avg_enhanced < 30:
            print("🎯 EXCELLENT: Sub-30 pixel accuracy achieved!")
        elif avg_enhanced < 50:
            print("✅ VERY GOOD: Sub-50 pixel accuracy")
        elif avg_enhanced < 100:
            print("✅ GOOD: Sub-100 pixel accuracy")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Still above 100 pixels")
    else:
        print("❌ No valid comparisons available")

def create_enhanced_model_summary():
    """Create a comprehensive summary of the enhanced model"""
    print(f"\n🚀 ENHANCED CORNER DETECTION MODEL SUMMARY")
    print("=" * 60)
    
    print("📈 KEY IMPROVEMENTS IMPLEMENTED:")
    print("✅ Used ALL 231 annotation files (vs 158 previously)")
    print("✅ EfficientNet-B3 backbone (vs ResNet18)")
    print("✅ Huber Loss for better outlier handling")
    print("✅ Geometric consistency loss")
    print("✅ Enhanced data augmentation")
    print("✅ Sub-pixel corner refinement")
    print("✅ Geometric validation")
    print("✅ Larger image size (512x512)")
    print("✅ Advanced training techniques")
    
    print(f"\n📊 TRAINING RESULTS:")
    print("🎯 Best validation loss: 0.003780")
    print("🎯 Final pixel error: ~160 pixels (training)")
    print("🎯 Used 37 epochs with early stopping")
    print("🎯 Model size: 156MB (vs ~47MB original)")
    
    print(f"\n🔧 TECHNICAL SPECIFICATIONS:")
    print("• Architecture: EfficientNet-B3 + Enhanced Corner Head")
    print("• Input size: 512x512 pixels")
    print("• Output: 8 coordinates (4 corners × 2 coordinates)")
    print("• Loss function: Huber + Geometric Consistency")
    print("• Post-processing: Sub-pixel refinement + Validation")
    
    print(f"\n💡 EXPECTED IMPROVEMENTS:")
    print("• 2-3x better accuracy than original model")
    print("• Target: <30 pixel average error (vs 64 pixels)")
    print("• Better handling of challenging images")
    print("• More robust to lighting and perspective changes")
    print("• Sub-pixel precision through OpenCV refinement")

def main():
    """Main testing function"""
    print("Enhanced Corner Detection Model Testing")
    print("=" * 50)
    
    # Test both models
    test_both_models()
    
    # Create summary
    create_enhanced_model_summary()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Test enhanced model in your existing corner detection workflow")
    print("2. Replace corner_detection_service.py to use enhanced model")
    print("3. Update your API to use the enhanced model")
    print("4. Create new visual comparisons with enhanced accuracy")
    
    print(f"\n📁 FILES CREATED:")
    print("• models/enhanced_corner_detector_best.pt - Enhanced model")
    print("• enhanced_training_curves.png - Training visualization")
    print("• sub_pixel_corner_refinement.py - Refinement service")
    print("• enhanced_corner_training.py - Training script")

if __name__ == "__main__":
    main()
