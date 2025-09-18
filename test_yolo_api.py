#!/usr/bin/env python3
"""
Test script for YOLO Corner Detection API
"""
import requests
import json

def test_yolo_api():
    api_url = "http://localhost:8002"
    
    print("🧪 Testing YOLO Corner Detection API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{api_url}/health")
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Model type: {health_data['model_type']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test root endpoint
    try:
        response = requests.get(f"{api_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            root_data = response.json()
            print(f"   Model loaded: {root_data['model_loaded']}")
            print(f"   Expected accuracy: {root_data['expected_accuracy']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test corner detection with a simple test
    print("\n🔍 Testing corner detection...")
    test_image_path = "my_chess_images/train/images/IMG_4698.JPG"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect_corners", files=files)
        
        print(f"   Corner detection response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"   Corners: {data['corners']}")
            print(f"   Processing time: {data['processing_time']}s")
            print(f"   Model: {data['model']}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text}")
                
    except Exception as e:
        print(f"❌ Corner detection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_api()
