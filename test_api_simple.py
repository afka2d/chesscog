#!/usr/bin/env python3
"""
Simple API test to isolate the issue.
"""

import requests
import json
from PIL import Image
import io

def test_api():
    """Test the API with a simple request."""
    print("🧪 Testing API with simple request")
    print("=" * 40)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Make API request
    files = {'image': ('test.jpg', image_data, 'image/jpeg')}
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    try:
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"   FEN: {result.get('fen', 'N/A')}")
            print(f"   Pieces detected: {len([p for p in result.get('pieces', []) if p is not None])}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n🎉 API is working!")
    else:
        print("\n❌ API has issues!")
