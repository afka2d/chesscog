#!/usr/bin/env python3
"""
Debug the real differences between Marshall and Original APIs
"""

import requests
import json
from pathlib import Path
import time

def test_api_with_real_image(api_url, image_path, corners):
    """Test an API with a real image"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'debug': True
            }
            
            start_time = time.time()
            response = requests.post(f"{api_url}/recognize_chess_position_with_corners", 
                                   files=files, data=data, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'pieces_found': result.get('pieces_found', 0),
                    'fen': result.get('fen', ''),
                    'processing_time': end_time - start_time,
                    'debug_info': result.get('debug_info', {})
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compare_apis_on_real_data():
    """Compare both APIs on real test data"""
    print("🧪 Comparing Marshall vs Original APIs on Real Data")
    print("=" * 60)
    
    # Test image and corners
    test_image = "yolo_detection_IMG_4763.jpg"
    test_corners = [[578.0, 1939.0], [2628.0, 1889.0], [2791.0, 4042.0], [397.0, 4025.0]]
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Using test image: {test_image}")
    print(f"📍 Using corners: {test_corners}")
    
    # Test both APIs
    apis = [
        ("Original API", "http://localhost:8001"),
        ("Marshall API", "http://localhost:8003")
    ]
    
    results = {}
    
    for api_name, api_url in apis:
        print(f"\n🔍 Testing {api_name}...")
        result = test_api_with_real_image(api_url, test_image, test_corners)
        results[api_name] = result
        
        if result['success']:
            print(f"   ✅ Success!")
            print(f"   📊 Pieces found: {result['pieces_found']}")
            print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"   🏁 FEN: {result['fen'][:50]}...")
            
            # Show debug info if available
            debug = result.get('debug_info', {})
            if debug:
                print(f"   🔧 Debug info:")
                print(f"      - Squares processed: {debug.get('squares_processed', 'N/A')}")
                print(f"      - Occupied squares: {debug.get('occupied_squares', 'N/A')}")
                print(f"      - Low confidence squares: {debug.get('low_confidence_squares', 'N/A')}")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    if results["Original API"]['success'] and results["Marshall API"]['success']:
        orig = results["Original API"]
        marshall = results["Marshall API"]
        
        print(f"Pieces Found:")
        print(f"   Original:  {orig['pieces_found']}")
        print(f"   Marshall:  {marshall['pieces_found']}")
        print(f"   Difference: {marshall['pieces_found'] - orig['pieces_found']}")
        
        print(f"\nProcessing Time:")
        print(f"   Original:  {orig['processing_time']:.2f}s")
        print(f"   Marshall:  {marshall['processing_time']:.2f}s")
        print(f"   Difference: {marshall['processing_time'] - orig['processing_time']:.2f}s")
        
        print(f"\nFEN Comparison:")
        print(f"   Original:  {orig['fen'][:50]}...")
        print(f"   Marshall:  {marshall['fen'][:50]}...")
        
        # Check if FENs are similar
        if orig['fen'] == marshall['fen']:
            print(f"   ✅ FENs are identical!")
        else:
            print(f"   ❌ FENs are different")
    
    print("\n" + "=" * 60)
    print("🎯 Real Data Comparison Complete!")

if __name__ == "__main__":
    compare_apis_on_real_data()
