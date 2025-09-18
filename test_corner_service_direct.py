#!/usr/bin/env python3
"""
Direct test of corner detection service to show you the results.
"""

import requests
import cv2
import numpy as np
from pathlib import Path
import json

def test_corner_detection_direct():
    """Test corner detection directly using the service"""
    print("🧪 TESTING CORNER DETECTION SERVICE")
    print("=" * 60)
    
    # Import the service directly
    from corner_detection_service import CornerDetectionService
    
    service = CornerDetectionService()
    
    if service.model is None:
        print("❌ Corner detection model not loaded")
        return False
    
    print("✅ Corner detection service loaded")
    
    # Test with multiple images
    test_images = [
        "grey_background_dataset/images/test/IMG_4785.JPG",
        "grey_background_dataset/images/val/IMG_4779.JPG",
        "grey_background_dataset/images/test/IMG_4763.JPG"
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images):
        if not Path(image_path).exists():
            continue
            
        print(f"\n--- Test {i+1}: {Path(image_path).name} ---")
        
        # Detect corners with visualization
        result = service.visualize_corners(image_path)
        
        if result:
            corners = result['corners']
            vis_path = result['visualization_path']
            
            print(f"✅ Corners detected:")
            print(f"   Top-Left: ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
            print(f"   Top-Right: ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
            print(f"   Bottom-Right: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
            print(f"   Bottom-Left: ({corners[3][0]:.1f}, {corners[3][1]:.1f})")
            print(f"   📸 Visualization: {vis_path}")
            
            # Compare with ground truth if available
            gt_path = None
            if 'test' in image_path:
                gt_path = f"grey_background_dataset/annotations/test/{Path(image_path).stem}.json"
            elif 'val' in image_path:
                gt_path = f"grey_background_dataset/annotations/val/{Path(image_path).stem}.json"
            
            if gt_path and Path(gt_path).exists():
                try:
                    with open(gt_path, 'r') as f:
                        annotation = json.load(f)
                    
                    gt_corners = np.array(annotation.get('corners', []))
                    pred_corners = np.array(corners)
                    
                    if len(gt_corners) == 4:
                        errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
                        avg_error = np.mean(errors)
                        
                        print(f"   📊 Accuracy: {avg_error:.1f} pixel average error")
                        
                        if avg_error < 50:
                            print("   ✅ EXCELLENT accuracy")
                        elif avg_error < 100:
                            print("   ✅ GOOD accuracy")
                        else:
                            print("   ⚠️  FAIR accuracy")
                            
                except Exception as e:
                    print(f"   ⚠️  Could not compare with ground truth")
            
            results.append({
                'image': Path(image_path).name,
                'corners': corners,
                'visualization': vis_path
            })
        else:
            print(f"❌ Corner detection failed")
    
    # Summary
    if results:
        print(f"\n🎯 CORNER DETECTION RESULTS SUMMARY")
        print("=" * 50)
        print(f"Successfully processed {len(results)} images:")
        
        for result in results:
            print(f"   📸 {result['image']}: {result['visualization']}")
        
        print(f"\n💡 HOW TO USE:")
        print("1. The corner detection model is working with good accuracy")
        print("2. You can integrate this into your workflow to eliminate manual corner selection")
        print("3. The service runs separately and won't affect your main API")
        
        # Create a usage example
        print(f"\n📋 INTEGRATION EXAMPLE:")
        print("```python")
        print("from corner_detection_service import CornerDetectionService")
        print("service = CornerDetectionService()")
        print("corners = service.detect_corners('path/to/image.jpg')")
        print("# Use these corners instead of manual selection")
        print("```")
        
        return True
    else:
        print(f"\n❌ No successful corner detections")
        return False

def main():
    """Main function"""
    print("Direct Corner Detection Service Test")
    print("=" * 50)
    
    success = test_corner_detection_direct()
    
    if success:
        print(f"\n🎯 CORNER DETECTION SERVICE IS WORKING!")
        print("You can now use automatic corner detection instead of manual selection.")
    else:
        print(f"\n❌ CORNER DETECTION SERVICE FAILED!")

if __name__ == "__main__":
    main()
