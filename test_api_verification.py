#!/usr/bin/env python3
"""
Test script to verify API changes are reflected in the app.
This will show all pieces as KINGS to prove the piece classifier is working.
"""

import requests
import json
from PIL import Image
import io

def test_api_verification():
    """Test the API to verify it's returning all KINGS."""
    print("🧪 API VERIFICATION TEST")
    print("=" * 50)
    print("This test will show all pieces as KINGS to prove the piece classifier is working.")
    print()
    
    # Test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Prepare the request
    url = "http://localhost:8000/recognize_chess_position_with_corners"
    
    with open(img_path, 'rb') as f:
        files = {'image': f}
        data = {
            'corners': json.dumps(corners),
            'color': 'white'
        }
        
        print("📡 Making API request...")
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            print("✅ API request successful!")
            print()
            
            # Analyze results
            pieces = result.get('pieces', [])
            occupied_pieces = [p for p in pieces if p is not None]
            
            print("📊 RESULTS:")
            print(f"   FEN: {result.get('fen', 'N/A')}")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Occupied squares: {len(occupied_pieces)}")
            print()
            
            # Check if all pieces are kings (K or k)
            king_count = sum(1 for p in occupied_pieces if p in ['K', 'k'])
            total_pieces = len(occupied_pieces)

            print("🧪 PIECE CLASSIFICATION VERIFICATION:")
            print(f"   Total pieces detected: {total_pieces}")
            print(f"   Kings detected: {king_count}")
            print(f"   King percentage: {(king_count/total_pieces*100) if total_pieces > 0 else 0:.1f}%")
            print()

            if king_count == total_pieces and total_pieces > 0:
                print("✅ SUCCESS: All pieces are KINGS! The piece classifier is working.")
                print("✅ This proves that API changes are being reflected in the app.")
            elif total_pieces == 0:
                print("⚠️  No pieces detected - this might be due to occupancy detection.")
            else:
                print("❌ FAILURE: Not all pieces are KINGS. The piece classifier might not be working.")
            
            print()
            print("🎯 PIECE BREAKDOWN:")
            piece_counts = {}
            for piece in occupied_pieces:
                piece_counts[piece] = piece_counts.get(piece, 0) + 1
            
            for piece, count in piece_counts.items():
                print(f"   {piece}: {count}")
            
            print()
            print("🏁 BOARD REPRESENTATION:")
            # Create a simple board representation
            board = [['.' for _ in range(8)] for _ in range(8)]
            for i, piece in enumerate(pieces):
                if piece is not None:
                    rank = 7 - (i // 8)  # Convert to chess notation
                    file = i % 8
                    if 'king' in piece.lower():
                        board[rank][file] = 'K' if 'white' in piece else 'k'
                    else:
                        board[rank][file] = 'P'  # Other pieces as P for now
            
            for row in board:
                print('   ' + ''.join(row))
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_verification()
