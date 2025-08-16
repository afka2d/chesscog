#!/usr/bin/env python3
"""
Diagnostic script to identify why the iOS app is unresponsive.
"""

import requests
import json
import time
import socket
from pathlib import Path

def diagnose_app_issue():
    """Diagnose the iOS app unresponsiveness issue."""
    
    print("=== iOS App Unresponsiveness Diagnosis ===")
    print()
    
    # Test 1: Basic network connectivity
    print("1. Testing basic network connectivity...")
    try:
        # Test DNS resolution
        ip = socket.gethostbyname("159.203.102.249")
        print(f"✅ DNS resolution: {ip}")
        
        # Test port connectivity
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(("159.203.102.249", 8000))
        sock.close()
        
        if result == 0:
            print("✅ Port 8000 is reachable")
        else:
            print(f"❌ Port 8000 is not reachable (error code: {result})")
            return
            
    except Exception as e:
        print(f"❌ Network connectivity failed: {e}")
        return
    
    # Test 2: Server health
    print("\n2. Testing server health...")
    try:
        health_response = requests.get("http://159.203.102.249:8000/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Server is healthy")
            print(f"   Models loaded: {health_data.get('models_loaded', 'Unknown')}")
            print(f"   Timestamp: {health_data.get('timestamp', 'Unknown')}")
        else:
            print(f"❌ Server responded with status: {health_response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Server health check failed: {e}")
        return
    
    # Test 3: Endpoint availability
    print("\n3. Testing endpoint availability...")
    try:
        # Test if endpoint exists
        response = requests.post(
            "http://159.203.102.249:8000/recognize_with_manual_corners",
            files={'image': ('test.jpg', b'fake_image_data', 'image/jpeg')},
            data={'corners': '[[0,0],[1,0],[1,1],[0,1]]', 'color': 'white'},
            timeout=5
        )
        
        if response.status_code == 400:
            print("✅ Endpoint exists (400 is expected for invalid image)")
        elif response.status_code == 200:
            print("✅ Endpoint exists and working")
        else:
            print(f"⚠️ Endpoint responded with unexpected status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ Endpoint request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Endpoint test failed: {e}")
    
    # Test 4: iOS app timeout simulation
    print("\n4. Testing iOS app timeout simulation...")
    print("   Simulating iOS app with 5-second timeout...")
    
    try:
        # Use a real test image
        image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
        if not Path(image_path).exists():
            print(f"   ⚠️ Test image not found: {image_path}")
            print("   Using minimal test instead...")
            
            # Minimal test
            response = requests.post(
                "http://159.203.102.249:8000/recognize_with_manual_corners",
                files={'image': ('test.jpg', b'fake_image_data', 'image/jpeg')},
                data={'corners': '[[0,0],[1,0],[1,1],[0,1]]', 'color': 'white'},
                timeout=5  # 5-second timeout like iOS app
            )
        else:
            # Real test with actual image
            corners = [
                [993, 2294],   # Top-left
                [2702, 2064],  # Top-right
                [2755, 3892],  # Bottom-right
                [542, 3864]    # Bottom-left
            ]
            
            files = {
                'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }
            
            data = {
                'corners': json.dumps(corners),
                'color': 'white'
            }
            
            start_time = time.time()
            response = requests.post(
                "http://159.203.102.249:8000/recognize_with_manual_corners",
                files=files,
                data=data,
                timeout=5  # 5-second timeout like iOS app
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                print(f"   ✅ Request completed in {processing_time:.2f}s")
                result = response.json()
                print(f"   Pieces found: {result.get('pieces_found', 'N/A')}")
                print(f"   FEN: {result.get('fen', 'N/A')}")
                
                if processing_time < 5:
                    print("   ✅ Response time is acceptable for iOS app")
                else:
                    print("   ⚠️ Response time might cause iOS app timeout")
            else:
                print(f"   ❌ Request failed with status: {response.status_code}")
                
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 5 seconds - this would cause iOS app to hang!")
        print("   💡 This is likely the cause of your app unresponsiveness")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 5: Alternative endpoints
    print("\n5. Testing alternative endpoints...")
    try:
        # Test the simple endpoint
        response = requests.get("http://159.203.102.249:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"⚠️ Root endpoint status: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 6: Recommendations
    print("\n6. Recommendations:")
    print("   If the endpoint times out (>5s):")
    print("   - The 3.5s processing time is too close to iOS timeout")
    print("   - Consider reducing image size or optimizing further")
    print("   - Add a loading indicator in the iOS app")
    print()
    print("   If network connectivity fails:")
    print("   - Check your device's internet connection")
    print("   - Verify the server IP is correct in the iOS app")
    print("   - Check if firewall is blocking the connection")
    print()
    print("   If endpoint doesn't exist:")
    print("   - Verify the iOS app is calling the correct URL")
    print("   - Check if the server was restarted recently")

if __name__ == "__main__":
    diagnose_app_issue() 