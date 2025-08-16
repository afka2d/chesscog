#!/usr/bin/env python3
"""
Test the API response timing and format to identify potential iOS parsing issues.
"""

import requests
import json
import time
from pathlib import Path

def test_response_timing():
    """Test response timing and format."""
    
    api_url = "https://api.chesspositionscanner.store/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    # Test with normalized coordinates (what iOS sends)
    normalized_corners = [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
    
    print("=== Testing Response Timing and Format ===")
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(normalized_corners),
            'color': 'white'
        }
        
        start_time = time.time()
        response = requests.post(api_url, files=files, data=data, timeout=60)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        print(f"📏 Content length: {len(response.content)} bytes")
        print(f"🔧 Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print("✅ JSON parsing successful")
                
                # Check each field
                print("\n📋 Response fields:")
                for key, value in json_data.items():
                    if key == 'debug_images':
                        if isinstance(value, dict):
                            print(f"  {key}: dict with {len(value)} images")
                            for img_key in value.keys():
                                img_data = value[img_key]
                                if isinstance(img_data, str):
                                    print(f"    {img_key}: {len(img_data)} chars")
                                else:
                                    print(f"    {img_key}: {type(img_data)}")
                        else:
                            print(f"  {key}: {type(value)}")
                    elif isinstance(value, list):
                        print(f"  {key}: list with {len(value)} items")
                    elif isinstance(value, str):
                        print(f"  {key}: \"{value[:50]}{'...' if len(value) > 50 else ''}\"")
                    else:
                        print(f"  {key}: {value}")
                
                # Test JSON serialization
                try:
                    serialized = json.dumps(json_data)
                    print(f"✅ Re-serialization successful: {len(serialized)} chars")
                except Exception as e:
                    print(f"❌ Re-serialization failed: {e}")
                    
                # Check for any problematic characters
                response_text = response.text
                problematic_chars = []
                for char in response_text:
                    if ord(char) > 127:  # Non-ASCII
                        if char not in problematic_chars:
                            problematic_chars.append(char)
                
                if problematic_chars:
                    print(f"⚠️  Found non-ASCII characters: {problematic_chars[:10]}")
                else:
                    print("✅ All characters are ASCII")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Response start: {response.text[:200]}")
                
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_response_timing()