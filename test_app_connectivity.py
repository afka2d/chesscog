#!/usr/bin/env python3
"""
Test script to check iOS app connectivity and endpoint responsiveness.
"""

import requests
import json
import time
from pathlib import Path

def test_app_connectivity():
    """Test if the iOS app can reach the server and endpoint."""
    
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
    
    print("=== Testing iOS App Connectivity ===")
    print(f"Server: {api_url}")
    print(f"Image: {image_path}")
    print(f"Corners: {corners}")
    print()
    
    # Test 1: Basic connectivity
    print("1. Testing basic server connectivity...")
    try:
        health_response = requests.get("http://159.203.102.249:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is reachable and healthy")
        else:
            print(f"❌ Server responded with status: {health_response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach server: {e}")
        return
    
    # Test 2: Endpoint responsiveness
    print("\n2. Testing endpoint responsiveness...")
    start_time = time.time()
    
    try:
        # Make the request with a shorter timeout to simulate iOS app behavior
        response = requests.post(api_url, files=files, data=data, timeout=15)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Response time: {processing_time:.2f} seconds")
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
            
            # Check if response time is acceptable for iOS app
            if processing_time < 5:
                print(f"✅ Response time ({processing_time:.2f}s) is acceptable for iOS app")
            else:
                print(f"⚠️ Response time ({processing_time:.2f}s) might be too slow for iOS app")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 15 seconds - this would cause iOS app to hang")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Network latency
    print("\n3. Testing network latency...")
    try:
        start_time = time.time()
        response = requests.get("http://159.203.102.249:8000/health", timeout=5)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Network latency: {latency:.1f}ms")
        
        if latency < 100:
            print("✅ Network latency is good")
        elif latency < 500:
            print("⚠️ Network latency is acceptable but could be better")
        else:
            print("❌ Network latency is poor - this could cause iOS app issues")
            
    except Exception as e:
        print(f"❌ Could not measure latency: {e}")

if __name__ == "__main__":
    test_app_connectivity() 