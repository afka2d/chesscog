#!/usr/bin/env python3
"""
Test the debug images in the API response to see if the chess board visualization is being generated.
"""

import requests
import json
import base64
from pathlib import Path

def test_debug_images():
    """Test if debug images are properly included in the API response."""
    
    api_url = "https://api.chesspositionscanner.store/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    # Test with normalized coordinates
    normalized_corners = [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
    
    print("=== Testing Debug Images ===")
    print(f"URL: {api_url}")
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(normalized_corners),
            'color': 'white'
        }
        
        response = requests.post(api_url, files=files, data=data, timeout=15)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            pieces_found = result.get('pieces_found', 0)
            fen = result.get('fen', 'N/A')
            debug_images = result.get('debug_images', {})
            
            print(f"✅ Pieces found: {pieces_found}")
            print(f"✅ FEN: {fen}")
            print(f"✅ Debug images keys: {list(debug_images.keys())}")
            
            # Check each debug image
            for image_name, image_data in debug_images.items():
                if image_data:
                    try:
                        # Try to decode the base64 image
                        decoded = base64.b64decode(image_data)
                        print(f"✅ {image_name}: {len(decoded)} bytes")
                    except Exception as e:
                        print(f"❌ {image_name}: Invalid base64 - {e}")
                else:
                    print(f"❌ {image_name}: Empty or None")
            
            # Specifically check for chess_board
            if 'chess_board' in debug_images:
                chess_board_data = debug_images['chess_board']
                if chess_board_data:
                    print("🎯 Chess board visualization is present!")
                    
                    # Save it to check
                    try:
                        decoded = base64.b64decode(chess_board_data)
                        with open('test_chess_board_output.png', 'wb') as f:
                            f.write(decoded)
                        print("💾 Saved chess board to test_chess_board_output.png")
                    except Exception as e:
                        print(f"❌ Failed to save chess board: {e}")
                else:
                    print("❌ Chess board visualization is empty/None")
            else:
                print("❌ No chess_board key in debug_images")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_debug_images()