#!/usr/bin/env python3
"""
Test the JSON response to see if there are any parsing issues.
"""

import requests
import json
from pathlib import Path

def test_json_response():
    """Test if the JSON response can be parsed properly."""
    
    api_url = "https://api.chesspositionscanner.store/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    # Test with normalized coordinates
    normalized_corners = [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
    
    print("=== Testing JSON Response Parsing ===")
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(normalized_corners),
            'color': 'white'
        }
        
        response = requests.post(api_url, files=files, data=data, timeout=30)
        
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        
        # Test JSON parsing
        try:
            json_data = response.json()
            print("✅ JSON parsing successful")
            
            # Check board_2d specifically
            board_2d = json_data.get('board_2d')
            if board_2d:
                print(f"✅ board_2d type: {type(board_2d)}")
                print(f"✅ board_2d length: {len(board_2d)}")
                print(f"✅ First row type: {type(board_2d[0])}")
                print(f"✅ First row length: {len(board_2d[0])}")
                print(f"✅ First element type: {type(board_2d[0][0])}")
                
                # Test JSON serialization again
                try:
                    json.dumps(board_2d)
                    print("✅ board_2d can be re-serialized")
                except Exception as e:
                    print(f"❌ board_2d serialization error: {e}")
                    
            else:
                print("❌ board_2d not found in response")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Request error: {e}")

if __name__ == "__main__":
    test_json_response()