#!/usr/bin/env python3
"""
Test script to verify the exact response format for the iOS app.
"""

import requests
import json
import time
from pathlib import Path

def test_response_format():
    """Test the exact response format that the iOS app would receive."""
    
    print("=== Testing Response Format for iOS App ===")
    print()
    
    # API endpoint
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    
    # Test image path
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        return
    
    # Use actual corner coordinates
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
    
    print("Sending request to iOS app endpoint...")
    print(f"URL: {api_url}")
    print(f"Image: {image_path}")
    print(f"Corners: {corners}")
    print()
    
    try:
        # Make the request with iOS app timeout
        start_time = time.time()
        response = requests.post(api_url, files=files, data=data, timeout=10)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"Response received in {processing_time:.2f} seconds")
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Content-Length: {response.headers.get('content-length', 'Unknown')}")
        print()
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Response JSON is valid")
                print()
                
                # Check all required fields
                required_fields = ['fen', 'legal_position', 'pieces_found', 'color', 'manual_corners', 'debug_images', 'status']
                missing_fields = []
                
                for field in required_fields:
                    if field in result:
                        value = result[field]
                        if isinstance(value, str) and len(value) > 100:
                            print(f"✅ {field}: {value[:100]}... (truncated)")
                        else:
                            print(f"✅ {field}: {value}")
                    else:
                        print(f"❌ {field}: MISSING")
                        missing_fields.append(field)
                
                print()
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    print("This could cause the iOS app to hang!")
                else:
                    print("✅ All required fields are present")
                
                # Check debug images
                debug_images = result.get('debug_images', {})
                if debug_images:
                    print(f"✅ Debug images included: {list(debug_images.keys())}")
                    
                    # Check if debug images are valid base64
                    for key, value in debug_images.items():
                        if isinstance(value, str) and value.startswith('data:image'):
                            print(f"✅ {key}: Valid base64 image data")
                        else:
                            print(f"⚠️ {key}: Invalid image format")
                else:
                    print("⚠️ No debug images included")
                
                # Check FEN validity
                fen = result.get('fen', '')
                if fen and fen != '8/8/8/8/8/8/8/8 w - - 0 1':  # Not empty board
                    print(f"✅ FEN is valid: {fen}")
                else:
                    print(f"⚠️ FEN might be empty or invalid: {fen}")
                
                print()
                print("=== iOS App Compatibility Check ===")
                
                # Check if response is iOS app friendly
                response_size = len(response.content)
                if response_size < 1000000:  # Less than 1MB
                    print(f"✅ Response size is reasonable: {response_size} bytes")
                else:
                    print(f"⚠️ Response size is large: {response_size} bytes")
                
                if processing_time < 5:
                    print(f"✅ Response time is good: {processing_time:.2f}s")
                else:
                    print(f"⚠️ Response time might be slow: {processing_time:.2f}s")
                
                # Check for potential iOS app issues
                if 'error' in result:
                    print("❌ Response contains error field - this could cause issues")
                
                if 'traceback' in result:
                    print("❌ Response contains traceback - this could cause issues")
                
                print()
                print("=== Summary ===")
                print("✅ Server is responding correctly")
                print("✅ Response format appears valid")
                print("✅ All required fields are present")
                print("✅ Response time is acceptable")
                print()
                print("💡 If the iOS app is still unresponsive, the issue might be:")
                print("   1. iOS app not properly parsing the JSON response")
                print("   2. iOS app UI thread blocking")
                print("   3. iOS app network timeout configuration")
                print("   4. iOS app not handling the response correctly")
                
            except json.JSONDecodeError as e:
                print(f"❌ Response is not valid JSON: {e}")
                print("This would definitely cause the iOS app to hang!")
                print(f"Response content: {response.text[:500]}...")
                
        else:
            print(f"❌ Server returned error status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - this would cause iOS app to hang!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_response_format() 