#!/usr/bin/env python3
"""
Test the API endpoint with the corrected transforms to verify it's working properly.
"""

import requests
import json
import base64
from pathlib import Path
import time

def test_api_endpoint():
    """Test the API endpoint with corrected transforms."""
    
    print("🧪 Testing API Endpoint with Corrected Transforms")
    print("=" * 60)
    
    # API endpoint
    api_url = "http://localhost:8002/recognize_chess_position_with_corners"
    
    # Test image path
    test_image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Test corner coordinates (approximate for IMG_4752)
    # These are example coordinates - you'll need to adjust based on your actual image
    test_corners = [[100, 100], [500, 100], [500, 500], [100, 500]]
    
    print(f"📁 Test image: {test_image_path}")
    print(f"🎯 Corner coordinates: {test_corners}")
    print(f"🌐 API endpoint: {api_url}")
    
    try:
        # Prepare the request
        files = {'image': open(test_image_path, 'rb')}
        data = {
            'corners': json.dumps(test_corners),
            'color': 'white',
            'debug_image_width': 800,
            'debug_image_height': 600
        }
        
        print(f"\n📤 Sending request to API...")
        start_time = time.time()
        
        # Make the request
        response = requests.post(api_url, files=files, data=data, timeout=60)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Response time: {processing_time:.2f} seconds")
        print(f"📊 HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API request successful!")
            
            # Parse response
            result = response.json()
            
            print(f"\n📋 Recognition Results:")
            print(f"  FEN: {result.get('fen', 'N/A')}")
            print(f"  Legal Position: {result.get('legal_position', 'N/A')}")
            print(f"  Pieces Found: {len(result.get('debug_images', {}))} debug images")
            
            # Check if debug images are present
            debug_images = result.get('debug_images', {})
            if debug_images:
                print(f"\n🖼️  Debug Images Available:")
                for key in debug_images.keys():
                    print(f"    - {key}")
            
            # Check processing info
            debug_info = result.get('debug_info', {})
            if debug_info:
                print(f"\n🔍 Processing Steps:")
                for step, status in debug_info.items():
                    print(f"    - {step}: {status}")
            
            print(f"\n💾 Full response saved to: api_test_response.json")
            
            # Save full response for inspection
            with open("api_test_response.json", "w") as f:
                json.dump(result, f, indent=2)
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error: Could not connect to API at {api_url}")
        print(f"Make sure the API server is running on port 8002")
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api_endpoint()

