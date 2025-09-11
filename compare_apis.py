#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image
import io

def test_both_apis():
    """Test both local and production APIs side by side"""
    print("🧪 Comparing Local vs Production APIs")
    print("=" * 50)
    
    # Test with real chess image
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    corners = [[0, 1260], [3240, 1260], [3240, 4500], [0, 4500]]
    turn = "white"
    
    print(f"📊 Testing with real chess image: {image_path}")
    print(f"📐 Using corners: {corners}")
    
    # Test local API
    print(f"\n🔍 Testing LOCAL API:")
    local_result = test_api("http://localhost:8000/recognize_chess_position_with_corners", image_path, corners, turn)
    
    # Test production API
    print(f"\n🔍 Testing PRODUCTION API:")
    production_result = test_api("https://api.chesspositionscanner.store/recognize_chess_position_with_corners", image_path, corners, turn)
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    if local_result and production_result:
        local_fen = local_result.get('fen', '')
        production_fen = production_result.get('fen', '')
        
        print(f"   Local API FEN:     {local_fen}")
        print(f"   Production API FEN: {production_fen}")
        
        if local_fen == production_fen:
            print("✅ Both APIs return identical FEN - they are working the same!")
        else:
            print("⚠️ APIs return different FEN - there may be a difference")
            
        # Check response formats
        print(f"\n📋 Response Format Comparison:")
        print(f"   Local API keys:     {list(local_result.keys())}")
        print(f"   Production API keys: {list(production_result.keys())}")
        
        # Check if production has additional fields
        if 'debug_image' in production_result:
            print("✅ Production API includes debug image")
        if 'debug_image_paths' in production_result:
            print("✅ Production API includes debug image paths")
        if 'corners' in production_result:
            print("✅ Production API includes corners in response")
        if 'processing_time' in production_result:
            print("✅ Production API includes processing time")
            
    else:
        print("❌ Could not compare - one or both APIs failed")

def test_api(api_url, image_path, corners, turn):
    """Test a single API"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'turn': turn
            }
            
            response = requests.post(api_url, files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response successful")
            print(f"📋 FEN: {result.get('fen', 'N/A')}")
            
            # Check if there are pieces detected
            fen = result.get('fen', '')
            if fen and fen != '8/8/8/8/8/8/8/8 w - - 0 1':
                print("✅ Pieces detected successfully!")
                
                # Count pieces in FEN
                piece_count = sum(1 for c in fen.split()[0] if c.isalpha())
                print(f"♟️ Total pieces detected: {piece_count}")
                
                # Show the board
                print("📋 Chess Board:")
                board_fen = fen.split()[0]
                ranks = board_fen.split('/')
                for i, rank in enumerate(ranks):
                    print(f"   {8-i} ", end="")
                    for char in rank:
                        if char.isdigit():
                            print("." * int(char), end="")
                        else:
                            print(char, end="")
                    print()
                print("     a b c d e f g h")
                
            else:
                print("❌ No pieces detected")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

if __name__ == "__main__":
    test_both_apis()
