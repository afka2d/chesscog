#!/usr/bin/env python3
"""
Test script for the new /recognize_with_manual_corners endpoint
"""

import requests
import json
import base64
from pathlib import Path

def test_manual_corners_endpoint():
    """Test the manual corners endpoint."""
    
    # API endpoint
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    
    # Test image path
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        return
    
    # Use actual corner coordinates from the annotation file
    corners = [
        [993, 2294],   # Top-left
        [2702, 2064],  # Top-right
        [2755, 3892],  # Bottom-right
        [542, 3864]    # Bottom-left
    ]
    
    # Prepare the request
    files = {
        'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    print("Testing /recognize_with_manual_corners endpoint...")
    print(f"Image: {image_path}")
    print(f"Corners: {corners}")
    
    try:
        # Make the request
        response = requests.post(api_url, files=files, data=data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint working successfully!")
            print(f"FEN: {result.get('fen', 'N/A')}")
            print(f"Legal position: {result.get('legal_position', 'N/A')}")
            print(f"Pieces found: {result.get('pieces_found', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            
            # Check if debug images are included
            debug_images = result.get('debug_images', {})
            if debug_images:
                print(f"Debug images included: {list(debug_images.keys())}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - this might indicate the hanging issue")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_manual_corners_endpoint() 