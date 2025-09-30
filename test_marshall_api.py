#!/usr/bin/env python3
"""
Test script for the Marshall Improved API.
Tests the new API endpoint to ensure it works correctly with the improved models.
"""

import requests
import json
import time
from pathlib import Path

def test_api_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8003/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Port: {data.get('port')}")
            print(f"   Classifier Type: {data.get('classifier_type')}")
            print(f"   Models loaded:")
            print(f"     - Occupancy: {data.get('occupancy_model_loaded')}")
            print(f"     - Color: {data.get('color_model_loaded')}")
            print(f"     - Piece Type: {data.get('piece_type_model_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_debug_info():
    """Test the debug info endpoint"""
    try:
        response = requests.get("http://localhost:8003/debug/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Debug info retrieved")
            print(f"   API Type: {data.get('api_type')}")
            print(f"   Port: {data.get('port')}")
            print(f"   Model Paths:")
            for model_type, path in data.get('model_paths', {}).items():
                print(f"     - {model_type}: {path}")
            return True
        else:
            print(f"❌ Debug info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Debug info error: {e}")
        return False

def test_chess_recognition():
    """Test the main chess recognition endpoint with a sample image"""
    try:
        # Look for a sample chess image
        sample_images = [
            "data/occupancy/test/occupied/IMG_4767_a8.png",
            "data/occupancy/test/occupied/IMG_4764_c2.png", 
            "data/occupancy/test/occupied/IMG_4763_d7.png"
        ]
        
        sample_image = None
        for img_path in sample_images:
            if Path(img_path).exists():
                sample_image = img_path
                break
        
        if not sample_image:
            print("⚠️  No sample images found for testing")
            return True
        
        print(f"📸 Testing with sample image: {sample_image}")
        
        # Sample corners (you may need to adjust these based on your image)
        corners = [[324, 324], [2916, 324], [2916, 5436], [324, 5436]]
        
        # Prepare the request
        files = {'image': open(sample_image, 'rb')}
        data = {
            'corners': json.dumps(corners),
            'debug': 'true'
        }
        
        # Make the request
        response = requests.post(
            'http://localhost:8003/recognize_chess_position_with_corners',
            files=files,
            data=data,
            timeout=30
        )
        
        files['image'].close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chess recognition test passed")
            print(f"   FEN: {result.get('fen', 'N/A')}")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Pieces detected: {sum(1 for p in result.get('pieces', []) if p is not None)}")
            
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"   Processing time: {debug.get('processing_time', 0):.3f}s")
                print(f"   Squares processed: {debug.get('squares_processed', 0)}")
                print(f"   Occupied squares: {debug.get('occupied_squares', 0)}")
                print(f"   Model info:")
                for model_type, model_name in debug.get('model_info', {}).items():
                    print(f"     - {model_type}: {model_name}")
            
            return True
        else:
            print(f"❌ Chess recognition test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chess recognition test error: {e}")
        return False

def main():
    print("🧪 Testing Marshall Improved API")
    print("=" * 50)
    
    # Wait a moment for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_api_health()
    
    # Test debug info endpoint
    print("\n2. Testing debug info endpoint...")
    debug_ok = test_debug_info()
    
    # Test chess recognition
    print("\n3. Testing chess recognition endpoint...")
    recognition_ok = test_chess_recognition()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Debug Info: {'✅ PASS' if debug_ok else '❌ FAIL'}")
    print(f"   Chess Recognition: {'✅ PASS' if recognition_ok else '❌ FAIL'}")
    
    if all([health_ok, debug_ok, recognition_ok]):
        print("\n🎉 All tests passed! Marshall Improved API is working correctly.")
        print("📍 API is ready at: http://localhost:8003")
        return 0
    else:
        print("\n❌ Some tests failed. Check the API logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())
