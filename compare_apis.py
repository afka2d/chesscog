#!/usr/bin/env python3
"""
Compare Original vs Marshall Improved APIs
Test both APIs on the same images to see performance differences
"""

import requests
import json
import time
from pathlib import Path
import cv2
import numpy as np

def test_api(api_url, image_path, api_name):
    """Test a single API endpoint"""
    print(f"\n🔍 Testing {api_name} API at {api_url}")
    
    try:
        # Read image
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            
            # Test corner detection
            start_time = time.time()
            response = requests.post(f"{api_url}/detect_corners", files=files, timeout=30)
            corner_time = time.time() - start_time
            
            if response.status_code == 200:
                corner_data = response.json()
                print(f"✅ Corner detection: {corner_time:.2f}s")
                print(f"   Corners: {corner_data.get('corners', 'N/A')}")
                print(f"   Confidence: {corner_data.get('confidence', 'N/A')}")
                
                # Test full analysis if corners detected
                if 'corners' in corner_data and corner_data['corners']:
                    corners_str = json.dumps(corner_data['corners'])
                    
                    start_time = time.time()
                    analysis_response = requests.post(
                        f"{api_url}/analyze_position",
                        files={'file': (image_path.name, open(image_path, 'rb'), 'image/jpeg')},
                        data={'corners': corners_str, 'fen': ''},
                        timeout=30
                    )
                    analysis_time = time.time() - start_time
                    
                    if analysis_response.status_code == 200:
                        analysis_data = analysis_response.json()
                        print(f"✅ Full analysis: {analysis_time:.2f}s")
                        print(f"   FEN: {analysis_data.get('fen', 'N/A')}")
                        print(f"   Occupancy count: {analysis_data.get('occupancy_count', 'N/A')}")
                        
                        return {
                            'api': api_name,
                            'corner_time': corner_time,
                            'analysis_time': analysis_time,
                            'total_time': corner_time + analysis_time,
                            'corners': corner_data.get('corners'),
                            'fen': analysis_data.get('fen'),
                            'occupancy_count': analysis_data.get('occupancy_count'),
                            'success': True
                        }
                    else:
                        print(f"❌ Analysis failed: {analysis_response.status_code}")
                        return {'api': api_name, 'success': False, 'error': 'Analysis failed'}
                else:
                    print("⚠️ No corners detected, skipping analysis")
                    return {'api': api_name, 'success': False, 'error': 'No corners detected'}
            else:
                print(f"❌ Corner detection failed: {response.status_code}")
                return {'api': api_name, 'success': False, 'error': 'Corner detection failed'}
                
    except Exception as e:
        print(f"❌ Error testing {api_name}: {e}")
        return {'api': api_name, 'success': False, 'error': str(e)}

def main():
    """Main comparison function"""
    print("🔄 API Comparison: Original vs Marshall Improved")
    print("=" * 60)
    
    # API endpoints
    original_api = "http://localhost:8001"
    marshall_api = "http://localhost:8006"
    
    # Test images
    test_images = [
        "marshall_chess_annotations/test_images/IMG_5851.HEIC",
        "marshall_chess_annotations/test_images/IMG_5852.HEIC",
        "marshall_chess_annotations/test_images/IMG_5853.HEIC"
    ]
    
    # Check if test images exist
    available_images = []
    for img_path in test_images:
        if Path(img_path).exists():
            available_images.append(img_path)
    
    if not available_images:
        print("❌ No test images found!")
        print("Please add some test images to marshall_chess_annotations/test_images/")
        return
    
    print(f"📸 Found {len(available_images)} test images")
    
    # Test each API
    results = []
    
    for image_path in available_images:
        print(f"\n📷 Testing image: {Path(image_path).name}")
        print("-" * 40)
        
        # Test original API
        original_result = test_api(original_api, image_path, "Original")
        results.append(original_result)
        
        # Test Marshall API
        marshall_result = test_api(marshall_api, image_path, "Marshall Improved")
        results.append(marshall_result)
    
    # Summary
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    original_results = [r for r in results if r['api'] == 'Original' and r['success']]
    marshall_results = [r for r in results if r['api'] == 'Marshall Improved' and r['success']]
    
    if original_results:
        avg_original_time = sum(r['total_time'] for r in original_results) / len(original_results)
        print(f"Original API - Average time: {avg_original_time:.2f}s")
        print(f"  Successful tests: {len(original_results)}")
    else:
        print("Original API - No successful tests")
    
    if marshall_results:
        avg_marshall_time = sum(r['total_time'] for r in marshall_results) / len(marshall_results)
        print(f"Marshall API - Average time: {avg_marshall_time:.2f}s")
        print(f"  Successful tests: {len(marshall_results)}")
    else:
        print("Marshall API - No successful tests")
    
    # Performance comparison
    if original_results and marshall_results:
        speed_improvement = ((avg_original_time - avg_marshall_time) / avg_original_time) * 100
        if speed_improvement > 0:
            print(f"🚀 Marshall API is {speed_improvement:.1f}% faster")
        else:
            print(f"⚠️ Marshall API is {abs(speed_improvement):.1f}% slower")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()