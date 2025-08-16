#!/usr/bin/env python3
"""
Test the production server (api.chesspositionscanner.store) to see if it has the coordinate conversion fix.
"""

import requests
import json
from pathlib import Path

def test_production_server():
    """Test the production server with normalized coordinates."""
    
    api_url = "https://api.chesspositionscanner.store/recognize_with_manual_corners"
    image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    
    # Test with normalized coordinates (what the iOS app sends)
    normalized_corners = [[0.3, 0.4], [0.8, 0.35], [0.85, 0.67], [0.16, 0.67]]
    
    print("=== Testing Production Server ===")
    print(f"URL: {api_url}")
    print(f"Normalized corners: {normalized_corners}")
    
    try:
        files = {
            'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'corners': json.dumps(normalized_corners),
            'color': 'white'
        }
        
        response = requests.post(api_url, files=files, data=data, timeout=15)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            pieces_found = result.get('pieces_found', 0)
            fen = result.get('fen', 'N/A')
            
            print(f"✅ Pieces found: {pieces_found}")
            print(f"✅ FEN: {fen}")
            
            if pieces_found > 0:
                print("🎉 SUCCESS: Production server has coordinate conversion working!")
            else:
                print("❌ ISSUE: Production server returning empty board with normalized coordinates")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_production_server()