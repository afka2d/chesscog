#!/usr/bin/env python3
"""
Test the API response structure to see what fields are returned.
"""

import requests
import json
from pathlib import Path

def test_api_response():
    """Test the API response structure."""
    
    api_url = "https://api.chesspositionscanner.store/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    # Test with normalized coordinates
    normalized_corners = [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
    
    print("=== Testing API Response Structure ===")
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(normalized_corners),
            'color': 'white'
        }
        
        response = requests.post(api_url, files=files, data=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            
            print("📋 Response fields:")
            for key, value in result.items():
                if key == 'debug_images':
                    print(f"  {key}: [dict with {len(value)} images]")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"  {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
            
            # Check specifically for board_2d
            if 'board_2d' in result:
                print(f"\n✅ board_2d found: {result['board_2d']}")
            else:
                print(f"\n❌ board_2d not found in response")
                
            # Check for similar fields
            board_fields = [k for k in result.keys() if 'board' in k.lower()]
            if board_fields:
                print(f"📋 Board-related fields: {board_fields}")
            else:
                print("❌ No board-related fields found")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_api_response()