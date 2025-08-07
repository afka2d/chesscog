#!/usr/bin/env python3
"""
Test script for the manual corners API endpoint.
Demonstrates how to submit manually corrected corner coordinates to the API.
"""

import requests
import json
import base64
import os
from pathlib import Path

def test_manual_corners():
    """Test the API with manually corrected corner coordinates."""
    
    print("=== Testing Manual Corners API ====")
    
    # Test image
    test_image = "IMG_4587.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Please ensure it exists in the current directory.")
        return
    
    # Example corner coordinates (you would get these from your iOS app)
    # These are example coordinates - you should replace with actual detected corners
    # Format: [[top_left], [top_right], [bottom_left], [bottom_right]]
    manual_corners = [
        [586.3321, 960.0475],   # Top-left
        [1109.8192, 978.328],   # Top-right  
        [584.748, 899.5496],    # Bottom-left
        [1109.9733, 982.7372]   # Bottom-right
    ]
    
    # API endpoint
    url = "http://localhost:8000/recognize_chess_position_with_corners"
    
    # Prepare the request
    files = {
        'image': (test_image, open(test_image, 'rb'), 'image/jpeg')
    }
    
    data = {
        'corners': json.dumps(manual_corners),
        'color': 'white',
        'debug_image_width': '800',
        'debug_image_height': '600'
    }
    
    print(f"Submitting image: {test_image}")
    print(f"Manual corners: {manual_corners}")
    
    try:
        # Make the request
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n=== SUCCESS ===")
            print(f"FEN: {result['fen']}")
            print(f"Legal Position: {result['legal_position']}")
            print(f"Lichess URL: {result['lichess_url']}")
            print(f"\nASCII Board:\n{result['ascii']}")
            
            print(f"\nDebug Images Generated:")
            for key, path in result.get('debug_image_paths', {}).items():
                print(f"  - {key}: {path}")
            
            print(f"\nDebug Images in Response:")
            for key in result.get('debug_images', {}).keys():
                print(f"  - {key}")
                
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_with_different_corners():
    """Test with slightly different corner coordinates to show the effect."""
    
    print("\n=== Testing with Different Corners ====")
    
    test_image = "IMG_4587.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found.")
        return
    
    # Slightly adjusted corner coordinates
    adjusted_corners = [
        [590.0, 965.0],    # Top-left (adjusted)
        [1115.0, 980.0],   # Top-right (adjusted)
        [580.0, 905.0],    # Bottom-left (adjusted)
        [1115.0, 985.0]    # Bottom-right (adjusted)
    ]
    
    url = "http://localhost:8000/recognize_chess_position_with_corners"
    
    files = {
        'image': (test_image, open(test_image, 'rb'), 'image/jpeg')
    }
    
    data = {
        'corners': json.dumps(adjusted_corners),
        'color': 'white',
        'debug_image_width': '800',
        'debug_image_height': '600'
    }
    
    print(f"Adjusted corners: {adjusted_corners}")
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"FEN: {result['fen']}")
            print(f"Legal Position: {result['legal_position']}")
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_manual_corners()
    test_with_different_corners() 