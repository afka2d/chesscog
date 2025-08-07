#!/usr/bin/env python3
"""
Test script to debug the custom model integration.
"""

import requests
import json

def test_custom_model():
    """Test the custom model with debug output."""
    
    url = "http://localhost:8002/recognize_chess_position_with_cursor_description"
    
    # Test with a training image
    with open("grey_background_dataset/images/train/IMG_4698.JPG", "rb") as f:
        files = {"image": ("IMG_4698.JPG", f, "image/jpeg")}
        data = {
            "cursor_description": "Test description for debugging",
            "color": "white"
        }
        
        print("Sending request to API...")
        response = requests.post(url, files=files, data=data)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"FEN: {result['fen']}")
            print(f"Pieces found: {result['pieces_found']}")
            print(f"Board 2D: {json.dumps(result['board_2d'], indent=2)}")
        else:
            print(f"Error response: {response.text}")

if __name__ == "__main__":
    test_custom_model() 