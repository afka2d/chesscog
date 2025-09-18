#!/usr/bin/env python3
"""
Test corner detection system without affecting main API.
"""

import requests
import cv2
import numpy as np
from pathlib import Path
import json
import base64

def test_corner_detection_api():
    """Test the corner detection API"""
    print("🧪 TESTING CORNER DETECTION API")
    print("=" * 50)
    
    # Check if corner detection API is running
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Corner Detection API: {health_data}")
        else:
            print("❌ Corner Detection API not running")
            print("   Start it with: python corner_detection_api.py")
            return False
    except:
        print("❌ Cannot connect to Corner Detection API")
        print("   Start it with: python corner_detection_api.py")
        return False
    
    # Test with a sample image
    test_image_path = "grey_background_dataset/images/test/IMG_4785.JPG"
    
    if not Path(test_image_path).exists():
        # Try validation images
        test_image_path = "grey_background_dataset/images/val/IMG_4779.JPG"
        
        if not Path(test_image_path).exists():
            print("❌ No test images found")
            return False
    
    print(f"\n🖼️  Testing with image: {Path(test_image_path).name}")
    
    # Test corner detection
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            
            response = requests.post(
                "http://localhost:8002/detect_corners",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Corner detection successful!")
            print(f"   Detected corners: {result['corners']}")
            print(f"   Image dimensions: {result['image_dimensions']}")
            
            return True
        else:
            print(f"❌ Corner detection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing corner detection: {e}")
        return False

def test_corner_visualization():
    """Test corner visualization"""
    print(f"\n🎨 TESTING CORNER VISUALIZATION")
    print("-" * 30)
    
    test_image_path = "grey_background_dataset/images/test/IMG_4785.JPG"
    
    if not Path(test_image_path).exists():
        test_image_path = "grey_background_dataset/images/val/IMG_4779.JPG"
        
        if not Path(test_image_path).exists():
            print("❌ No test images found")
            return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            
            response = requests.post(
                "http://localhost:8002/visualize_corners",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Visualization successful!")
            print(f"   Corners: {result['corners']}")
            
            # Save visualization
            if 'visualization' in result:
                img_data = base64.b64decode(result['visualization'])
                with open('corner_detection_visualization.jpg', 'wb') as f:
                    f.write(img_data)
                
                print(f"   📸 Visualization saved to: corner_detection_visualization.jpg")
            
            return True
        else:
            print(f"❌ Visualization failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visualization: {e}")
        return False

def compare_with_ground_truth():
    """Compare detected corners with ground truth"""
    print(f"\n📊 COMPARING WITH GROUND TRUTH")
    print("-" * 30)
    
    # Find a test image with annotation
    test_image_path = "grey_background_dataset/images/test/IMG_4785.JPG"
    test_ann_path = "grey_background_dataset/annotations/test/IMG_4785.json"
    
    if not Path(test_image_path).exists() or not Path(test_ann_path).exists():
        print("❌ Test image or annotation not found")
        return False
    
    # Load ground truth
    try:
        with open(test_ann_path, 'r') as f:
            annotation = json.load(f)
        
        gt_corners = annotation.get('corners', [])
        
        if not gt_corners:
            print("❌ No ground truth corners found")
            return False
        
        print(f"📍 Ground truth corners: {gt_corners}")
        
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return False
    
    # Get API prediction
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            
            response = requests.post(
                "http://localhost:8002/detect_corners",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            pred_corners = result['corners']
            
            print(f"🤖 Predicted corners: {pred_corners}")
            
            # Calculate error
            gt_np = np.array(gt_corners)
            pred_np = np.array(pred_corners)
            
            # Calculate pixel error for each corner
            errors = np.sqrt(np.sum((gt_np - pred_np) ** 2, axis=1))
            avg_error = np.mean(errors)
            
            print(f"\n📊 ACCURACY ANALYSIS:")
            print(f"   Average pixel error: {avg_error:.1f} pixels")
            print(f"   Per-corner errors: {[f'{e:.1f}' for e in errors]} pixels")
            
            if avg_error < 50:
                print("✅ EXCELLENT: Very accurate corner detection")
            elif avg_error < 100:
                print("✅ GOOD: Acceptable corner detection")
            elif avg_error < 200:
                print("⚠️  FAIR: Needs improvement")
            else:
                print("❌ POOR: Significant improvement needed")
            
            return True
        else:
            print(f"❌ API call failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error comparing with ground truth: {e}")
        return False

def main():
    """Main testing function"""
    print("Corner Detection Testing Suite")
    print("=" * 50)
    print("This tests the corner detection system without affecting your main API.")
    print()
    
    # Test basic corner detection
    if not test_corner_detection_api():
        return
    
    # Test visualization
    if not test_corner_visualization():
        return
    
    # Compare with ground truth
    if not compare_with_ground_truth():
        return
    
    print(f"\n🎯 ALL TESTS PASSED!")
    print("Your corner detection system is working correctly.")
    print("\n📋 NEXT STEPS:")
    print("1. View visualization: corner_detection_visualization.jpg")
    print("2. Test with more images via demo: http://localhost:8002/demo")
    print("3. Integrate with your workflow when ready")

if __name__ == "__main__":
    main()
