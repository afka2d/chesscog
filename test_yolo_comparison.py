#!/usr/bin/env python3
"""
Script to compare YOLO versions and show results
"""

import requests
import json
import time
from pathlib import Path

def test_yolo_upgrade(image_path: str, test_api_url: str = "http://localhost:8012"):
    """
    Test YOLO upgrade on a specific image
    """
    print(f"🎯 Testing YOLO upgrades on: {image_path}")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Test each version
    versions_to_test = ["yolov8", "yolov9", "yolov11"]
    results = {}
    
    for version in versions_to_test:
        print(f"\n🔍 Testing {version}...")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{test_api_url}/test/{version}", files=files)
                
            if response.status_code == 200:
                result = response.json()
                results[version] = result
                
                if result['success']:
                    print(f"   ✅ Success: {result['processing_time']:.3f}s, confidence: {result['debug_info'].get('confidence', 'N/A')}")
                    print(f"   📍 Corners: {len(result['corners']) if result['corners'] else 0} detected")
                else:
                    print(f"   ❌ Failed: {result.get('debug_info', {}).get('error', 'Unknown error')}")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to test API. Make sure it's running on {test_api_url}")
            print(f"   💡 Start it with: python3 test_yolo_upgrade_api.py")
            return
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ No successful detections")
        return
    
    # Find fastest
    fastest = min(successful_results.items(), key=lambda x: x[1]['processing_time'])
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['processing_time']:.3f}s)")
    
    # Find highest confidence
    highest_conf = max(successful_results.items(), 
                      key=lambda x: x[1]['debug_info'].get('confidence', 0))
    print(f"🎯 Highest Confidence: {highest_conf[0]} ({highest_conf[1]['debug_info'].get('confidence', 'N/A')})")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for version, result in successful_results.items():
        print(f"\n{version.upper()}:")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print(f"   Confidence: {result['debug_info'].get('confidence', 'N/A')}")
        print(f"   Corners: {result['corners']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if 'yolov9' in successful_results and 'yolov8' in successful_results:
        v8_time = successful_results['yolov8']['processing_time']
        v9_time = successful_results['yolov9']['processing_time']
        v8_conf = successful_results['yolov8']['debug_info'].get('confidence', 0)
        v9_conf = successful_results['yolov9']['debug_info'].get('confidence', 0)
        
        if v9_conf > v8_conf:
            improvement = ((v9_conf - v8_conf) / v8_conf) * 100
            print(f"   🎯 YOLOv9 shows {improvement:.1f}% confidence improvement over YOLOv8")
        
        if v9_time <= v8_time * 1.2:  # Within 20% speed
            print(f"   ⚡ YOLOv9 speed is acceptable (within 20% of YOLOv8)")
            print(f"   🚀 RECOMMENDED: Upgrade to YOLOv9")
        else:
            print(f"   ⚠️  YOLOv9 is {((v9_time - v8_time) / v8_time) * 100:.1f}% slower than YOLOv8")
    
    if 'yolov11' in successful_results:
        v11_conf = successful_results['yolov11']['debug_info'].get('confidence', 0)
        if v11_conf > 0.8:
            print(f"   🏆 YOLOv11 shows excellent confidence ({v11_conf:.2f})")
            print(f"   🎯 Consider YOLOv11 for maximum accuracy")

def main():
    """Main function"""
    print("🎯 YOLO Upgrade Comparison Tool")
    print("=" * 40)
    
    # Test with your chess images
    test_images = [
        "yolo_detection_IMG_4763.jpg",
        "id.jpg",
        "IMG_6904.jpg"  # From your desktop
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            test_yolo_upgrade(image_path)
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"⚠️  Skipping {image_path} (not found)")
    
    print("✅ Testing complete!")
    print("\n💡 To test manually:")
    print("   1. Start test API: python3 test_yolo_upgrade_api.py")
    print("   2. Visit: http://localhost:8012/demo")
    print("   3. Upload images and compare results")

if __name__ == "__main__":
    main()

