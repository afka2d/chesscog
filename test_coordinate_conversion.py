#!/usr/bin/env python3
"""
Test the coordinate conversion specifically.
"""

import requests
import json
from pathlib import Path

def test_coordinate_conversion():
    """Test coordinate conversion with various formats."""
    
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    test_cases = [
        {
            "name": "Perfect normalized (0-1)",
            "corners": [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
        },
        {
            "name": "Slightly out of bounds (should still convert)",
            "corners": [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]]
        },
        {
            "name": "Clearly pixel coordinates",
            "corners": [[993, 2294], [2702, 2064], [2755, 3892], [542, 3864]]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        print(f"Corners: {test_case['corners']}")
        
        try:
            files = {
                'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }
            
            data = {
                'corners': json.dumps(test_case['corners']),
                'color': 'white'
            }
            
            response = requests.post(api_url, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                pieces_found = result.get('pieces_found', 0)
                print(f"Result: {pieces_found} pieces found")
                print(f"Status: {'✅ Success' if pieces_found > 0 else '❌ Empty board'}")
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_coordinate_conversion()