#!/usr/bin/env python3
"""
Test script for the Chess Position Scanner API
"""
import requests
import json
import base64
from pathlib import Path

def test_api_health(base_url="http://localhost:8000"):
    """Test the health endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status')}")
            print(f"Models loaded: {data.get('models_loaded')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_recognition(image_path, base_url="http://localhost:8000"):
    """Test the recognition endpoint."""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'color': 'white'}
            
            print(f"Testing recognition with {image_path}...")
            response = requests.post(
                f"{base_url}/recognize_chess_position",
                files=files,
                data=data,
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"FEN: {result.get('fen')}")
                print(f"Legal position: {result.get('legal_position')}")
                print(f"Lichess URL: {result.get('lichess_url')}")
                
                # Check debug images
                debug_images = result.get('debug_images', {})
                print(f"Debug images available: {list(debug_images.keys())}")
                
                # Save debug images if available
                for key, img_data in debug_images.items():
                    if img_data:
                        img_bytes = base64.b64decode(img_data)
                        output_path = f"debug_{key}.png"
                        with open(output_path, 'wb') as img_file:
                            img_file.write(img_bytes)
                        print(f"Saved debug image: {output_path}")
                
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"Recognition test failed: {e}")
        return False

def test_corner_detection(image_path, base_url="http://localhost:8000"):
    """Test the corner detection endpoint."""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            
            print(f"Testing corner detection with {image_path}...")
            response = requests.post(
                f"{base_url}/detect_corners",
                files=files,
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                corners = result.get('corners')
                print(f"Corners detected: {len(corners) if corners else 0}")
                
                # Check debug images
                debug_images = result.get('debug_images', {})
                print(f"Debug images available: {list(debug_images.keys())}")
                
                # Save debug images if available
                for key, img_data in debug_images.items():
                    if img_data:
                        img_bytes = base64.b64decode(img_data)
                        output_path = f"corner_debug_{key}.png"
                        with open(output_path, 'wb') as img_file:
                            img_file.write(img_bytes)
                        print(f"Saved debug image: {output_path}")
                
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"Corner detection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Chess Position Scanner API Test ===\n")
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    if not test_api_health():
        print("Health check failed. Make sure the API is running.")
        return
    
    print("\n2. Testing corner detection...")
    # Test with a sample image
    test_images = [
        "IMG_4540.jpeg",
        "debug_cropped_board.jpeg",
        "sample.jpeg"
    ]
    
    for img in test_images:
        if Path(img).exists():
            test_corner_detection(img)
            break
    
    print("\n3. Testing recognition...")
    for img in test_images:
        if Path(img).exists():
            test_recognition(img)
            break
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main() 