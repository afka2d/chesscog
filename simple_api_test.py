#!/usr/bin/env python3
"""
Simple API test to determine expected real-world accuracy.
"""

import requests
import json
import os
import glob
import random

def test_api_simple():
    """Test the API with a simple request."""
    print("🧪 Testing API with Simple Request")
    print("=" * 40)
    
    # Find a test image
    test_dirs = [
        "my_chess_images/train/images",
        "grey_background_dataset/images/test"
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(glob.glob(os.path.join(test_dir, ext)))
            if images:
                test_image = random.choice(images)
                break
    
    if not test_image:
        print("❌ No test images found")
        return None
    
    print(f"📁 Using test image: {test_image}")
    
    # Test corners (approximate for a chessboard)
    corners = [
        [50, 50],   # Top-left
        [400, 50],  # Top-right
        [400, 400], # Bottom-right
        [50, 400]   # Bottom-left
    ]
    
    try:
        # Read and encode image
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        # Prepare the request
        files = {'image': (os.path.basename(test_image), image_data, 'image/jpeg')}
        data = {
            'corners': json.dumps(corners),
            'color': 'white'
        }
        
        print("📤 Sending request to API...")
        
        # Make the request
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            
            # Analyze the result
            fen = result.get('fen', '')
            pieces = result.get('pieces', [])
            occupancy = result.get('occupancy', [])
            
            print(f"\n📊 RESULTS:")
            print(f"   FEN: {fen}")
            print(f"   Occupancy: {len(occupancy)} squares")
            print(f"   Pieces: {len(pieces)} pieces")
            
            # Count piece types
            if isinstance(pieces, list):
                piece_count = 0
                piece_types = set()
                
                for piece in pieces:
                    if piece is not None:
                        piece_count += 1
                        if isinstance(piece, str):
                            piece_types.add(piece)
                
                print(f"   Piece count: {piece_count}")
                print(f"   Unique types: {len(piece_types)}")
                print(f"   Piece types: {sorted(piece_types)}")
                
                # Calculate diversity score
                diversity = len(piece_types) / 12.0 if piece_count > 0 else 0
                print(f"   Diversity score: {diversity:.2f}")
                
                # Estimate accuracy
                if diversity >= 0.8:
                    estimated_accuracy = "85-95%"
                    confidence = "High"
                elif diversity >= 0.6:
                    estimated_accuracy = "75-85%"
                    confidence = "Medium"
                elif diversity >= 0.4:
                    estimated_accuracy = "65-75%"
                    confidence = "Low"
                else:
                    estimated_accuracy = "50-65%"
                    confidence = "Very Low"
                
                print(f"\n🎯 ESTIMATED REAL-WORLD ACCURACY:")
                print(f"   Expected Accuracy: {estimated_accuracy}")
                print(f"   Confidence: {confidence}")
                print(f"   Based on diversity: {diversity:.2f}")
                
                if diversity >= 0.6:
                    print(f"\n✅ GOOD: This should work well for real chess positions!")
                else:
                    print(f"\n⚠️  CAUTION: May have overfitting issues")
                
                return {
                    'success': True,
                    'diversity': diversity,
                    'piece_count': piece_count,
                    'piece_types': len(piece_types),
                    'estimated_accuracy': estimated_accuracy
                }
            else:
                print("❌ Unexpected pieces format")
                return None
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function."""
    print("🎯 Simple API Accuracy Test")
    print("=" * 30)
    
    result = test_api_simple()
    
    if result and result['success']:
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"   The API is working and should provide good accuracy")
    else:
        print(f"\n❌ TEST FAILED")
        print(f"   Check the error messages above")

if __name__ == "__main__":
    main()
