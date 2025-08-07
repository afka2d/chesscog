#!/usr/bin/env python3
"""
Test script for the manual corners API endpoint with custom corner coordinates.
This allows you to easily test different corner coordinates for your chess board.
"""

import requests
import json
import base64
import os
from pathlib import Path

def test_custom_corners(image_path, corners, color="white"):
    """
    Test the API with custom corner coordinates.
    
    Args:
        image_path: Path to the chess board image
        corners: List of 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        color: "white" or "black"
    """
    
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return False
    
    print(f"=== Testing Custom Corners ===")
    print(f"Image: {image_path}")
    print(f"Corners: {corners}")
    print(f"Color: {color}")
    print()
    
    # API endpoint
    url = "http://localhost:8001/recognize_chess_position_with_corners"
    
    # Prepare the request
    files = {
        'image': (image_path, open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'corners': json.dumps(corners),
        'color': color,
        'debug_image_width': '800',
        'debug_image_height': '600'
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("=== SUCCESS ===")
            print(f"FEN: {result['fen']}")
            print(f"Legal Position: {result['legal_position']}")
            print(f"Lichess URL: {result['lichess_url']}")
            print()
            print("ASCII Board:")
            print(result['ascii'])
            print()
            
            # Check debug images
            debug_images = result.get('debug_images', {})
            debug_image_paths = result.get('debug_image_paths', {})
            
            print("Debug Images Generated:")
            for key, path in debug_image_paths.items():
                print(f"  - {key}: {path}")
            
            print("\nDebug Images in Response:")
            for key in debug_images.keys():
                print(f"  - {key}")
            
            # Save board_focus image with custom name
            if 'board_focus' in debug_images:
                img_data = base64.b64decode(debug_images['board_focus'])
                custom_filename = f"custom_corners_board_focus_{len(corners)}.png"
                with open(custom_filename, 'wb') as f:
                    f.write(img_data)
                print(f"\nSaved custom board focus image: {custom_filename}")
            
            return True
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def interactive_corner_test():
    """Interactive mode to test different corner coordinates."""
    
    print("=== Interactive Corner Testing ===")
    
    # Get image path
    image_path = input("Enter image path (or press Enter for IMG_4587.jpg): ").strip()
    if not image_path:
        image_path = "IMG_4587.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        return
    
    print(f"Using image: {image_path}")
    print()
    
    # Get color
    color = input("Enter color (white/black, default: white): ").strip().lower()
    if color not in ["white", "black"]:
        color = "white"
    
    while True:
        print("\n" + "="*50)
        print("Enter corner coordinates (or 'quit' to exit):")
        print("Format: x1,y1 x2,y2 x3,y3 x4,y4")
        print("Example: 100,200 300,200 300,400 100,400")
        
        user_input = input("Corners: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        try:
            # Parse corner coordinates
            corner_strs = user_input.split()
            if len(corner_strs) != 4:
                print("Error: Need exactly 4 corner coordinates")
                continue
            
            corners = []
            for corner_str in corner_strs:
                x, y = map(float, corner_str.split(','))
                corners.append([x, y])
            
            print(f"Parsed corners: {corners}")
            
            # Test the corners
            test_custom_corners(image_path, corners, color)
            
        except ValueError as e:
            print(f"Error parsing coordinates: {e}")
            print("Please use format: x1,y1 x2,y2 x3,y3 x4,y4")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        if len(sys.argv) < 6:
            print("Usage: python test_custom_corners.py <image> <x1,y1> <x2,y2> <x3,y3> <x4,y4> [color]")
            print("Example: python test_custom_corners.py IMG_4587.jpg 100,200 300,200 300,400 100,400 white")
            sys.exit(1)
        
        image_path = sys.argv[1]
        corners = []
        for i in range(2, 6):
            x, y = map(float, sys.argv[i].split(','))
            corners.append([x, y])
        
        color = sys.argv[6] if len(sys.argv) > 6 else "white"
        
        test_custom_corners(image_path, corners, color)
    else:
        # Interactive mode
        interactive_corner_test() 