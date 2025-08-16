#!/usr/bin/env python3
"""
Debug script to analyze the format and properties of app photos
that are causing occupancy detection to fail.
"""

import requests
import numpy as np
import cv2
import json
import base64
from pathlib import Path

def analyze_recent_request():
    """Analyze the most recent request that failed."""
    
    print("=== Analyzing App Photo Issues ===")
    
    # Create a test with the exact format the app would send
    test_image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Load and analyze the test image
    img = cv2.imread(test_image_path)
    print(f"Test image properties:")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.dtype}")
    print(f"  Min/Max values: {img.min()}/{img.max()}")
    print(f"  File size: {Path(test_image_path).stat().st_size} bytes")
    
    # Test normalized vs non-normalized corners
    print("\n--- Testing Corner Formats ---")
    
    # Original corners (pixel coordinates)
    pixel_corners = [[993, 2294], [2702, 2064], [2755, 3892], [542, 3864]]
    print(f"Pixel corners: {pixel_corners}")
    
    # Normalized corners (like iOS might send)
    img_height, img_width = img.shape[:2]
    normalized_corners = [
        [corner[0] / img_width, corner[1] / img_height] 
        for corner in pixel_corners
    ]
    print(f"Normalized corners: {normalized_corners}")
    
    # Test both formats with the API
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    
    print("\n--- Testing Pixel Corners ---")
    test_api_with_corners(api_url, test_image_path, pixel_corners)
    
    print("\n--- Testing Normalized Corners ---")
    test_api_with_corners(api_url, test_image_path, normalized_corners)
    
def test_api_with_corners(api_url, image_path, corners):
    """Test the API with specific corner format."""
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(corners),
            'color': 'white'
        }
        
        response = requests.post(api_url, files=files, data=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            pieces_found = result.get('pieces_found', 0)
            fen = result.get('fen', 'N/A')
            
            print(f"  Result: {pieces_found} pieces found")
            print(f"  FEN: {fen}")
            print(f"  {'✅ Success' if pieces_found > 0 else '❌ Empty board'}")
        else:
            print(f"  ❌ API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"  ❌ Request failed: {e}")

def create_test_scenarios():
    """Create test scenarios to identify the issue."""
    
    print("\n=== Creating Test Scenarios ===")
    
    # Scenario 1: Very small corners (like mobile might send)
    small_corners = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    print(f"Small normalized corners: {small_corners}")
    
    # Scenario 2: Invalid corners 
    invalid_corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    print(f"Invalid corners: {invalid_corners}")
    
    # Scenario 3: Out of bounds corners
    oob_corners = [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]]
    print(f"Out of bounds corners: {oob_corners}")
    
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    test_image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    scenarios = [
        ("Small normalized", small_corners),
        ("Invalid (unit square)", invalid_corners), 
        ("Out of bounds", oob_corners)
    ]
    
    for name, corners in scenarios:
        print(f"\n--- Testing {name} ---")
        test_api_with_corners(api_url, test_image_path, corners)

if __name__ == "__main__":
    analyze_recent_request()
    create_test_scenarios()