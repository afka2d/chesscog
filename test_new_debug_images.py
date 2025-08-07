#!/usr/bin/env python3
"""
Test script for the updated Chess Position Scanner API with new debug images.
Tests the new debug images from the recognition pipeline including occupancy_map and piece_map.
"""

import requests
import base64
import json
import os
from pathlib import Path

def test_new_debug_images():
    """Test the API with new debug images from the recognition pipeline."""
    
    print("=== Testing New Debug Images API ===")
    
    # Test image
    test_image = "IMG_4587.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found. Please ensure it exists in the current directory.")
        return
    
    # API endpoint
    url = "http://localhost:8000/recognize_chess_position"
    
    print(f"1. Testing recognition with new debug images using {test_image}...")
    
    # Prepare the request
    with open(test_image, 'rb') as f:
        files = {'image': (test_image, f, 'image/jpeg')}
        data = {
            'color': 'white',
            'debug_image_width': 800,
            'debug_image_height': 600
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"Response status: {response.status_code}")
                print(f"FEN: {result.get('fen', 'N/A')}")
                print(f"Legal position: {result.get('legal_position', 'N/A')}")
                print(f"Lichess URL: {result.get('lichess_url', 'N/A')}")
                
                # Check for new debug images
                debug_images = result.get('debug_images', {})
                print(f"\nDebug images available: {list(debug_images.keys())}")
                
                # Check for the new debug images we added
                new_debug_images = ['warped_board', 'occupancy_map', 'piece_map']
                found_new_images = []
                missing_new_images = []
                
                for img_name in new_debug_images:
                    if img_name in debug_images:
                        found_new_images.append(img_name)
                    else:
                        missing_new_images.append(img_name)
                
                print(f"\nNew debug images found: {found_new_images}")
                if missing_new_images:
                    print(f"Missing new debug images: {missing_new_images}")
                
                # Save all debug images
                print(f"\nSaving debug images...")
                for img_name, img_data in debug_images.items():
                    if img_data:
                        try:
                            # Decode base64 image
                            img_bytes = base64.b64decode(img_data)
                            
                            # Save to file
                            filename = f"new_debug_{img_name}.png"
                            with open(filename, 'wb') as img_file:
                                img_file.write(img_bytes)
                            print(f"Saved debug image: {filename}")
                        except Exception as e:
                            print(f"Failed to save {img_name}: {e}")
                
                # Check processing info
                debug_info = result.get('debug_info', {})
                print(f"\nDebug info: {debug_info}")
                
                # Check image info
                image_info = result.get('image_info', {})
                print(f"Image info: {image_info}")
                
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

def test_corner_detection():
    """Test the corner detection endpoint for comparison."""
    
    print(f"\n2. Testing corner detection for comparison...")
    
    test_image = "IMG_4587.jpg"
    url = "http://localhost:8000/detect_corners"
    
    with open(test_image, 'rb') as f:
        files = {'image': (test_image, f, 'image/jpeg')}
        
        try:
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                debug_images = result.get('debug_images', {})
                print(f"Corner detection debug images: {list(debug_images.keys())}")
            else:
                print(f"Corner detection error: {response.status_code}")
                
        except Exception as e:
            print(f"Corner detection test failed: {e}")

if __name__ == "__main__":
    test_new_debug_images()
    test_corner_detection()
    print("\n=== Test completed ===")
    print("Check the generated debug images to see the new recognition pipeline debug images!") 