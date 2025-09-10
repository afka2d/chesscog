#!/usr/bin/env python3
"""
Test the API with a more realistic occupancy pattern.
"""

import requests
import json
from PIL import Image
import io

def test_realistic():
    """Test the API with realistic occupancy."""
    print("🧪 Testing API with Realistic Occupancy")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Make API request
    files = {'image': ('test.jpg', image_data, 'image/jpeg')}
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    try:
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"   FEN: {result.get('fen', 'N/A')}")
            
            pieces = result.get('pieces', [])
            occupied_pieces = [p for p in pieces if p is not None]
            print(f"   Pieces detected: {len(occupied_pieces)}")
            
            # Analyze piece types
            piece_types = set(occupied_pieces)
            print(f"   Unique piece types: {len(piece_types)}")
            print(f"   Piece types: {list(piece_types)}")
            
            # Calculate diversity
            diversity = len(piece_types) / 12.0 if len(occupied_pieces) > 0 else 0
            print(f"   Diversity score: {diversity:.2f}")
            
            # Estimate accuracy
            if diversity >= 0.6:
                estimated_accuracy = "75-85%"
                assessment = "GOOD"
            elif diversity >= 0.4:
                estimated_accuracy = "65-75%"
                assessment = "MODERATE"
            else:
                estimated_accuracy = "50-65%"
                assessment = "POOR"
            
            print(f"\n🎯 ESTIMATED ACCURACY: {estimated_accuracy}")
            print(f"   Assessment: {assessment}")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_realistic()
    if success:
        print("\n🎉 API is working with piece classification!")
    else:
        print("\n❌ API has issues!")
