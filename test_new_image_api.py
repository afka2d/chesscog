#!/usr/bin/env python3
"""
Test script to call the API with the new attached image.
This tests the /recognize_chess_position_with_cursor_description endpoint.
"""

import requests
import json
import time
import os

def test_api_with_new_image():
    """Test the API with the new attached image."""
    
    # The Cursor description from the new image
    cursor_description = """
    This image displays a chess board with several pieces, viewed from a slightly elevated angle, resting on a light-colored wooden surface.

    **High-Level Description:**
    The image shows a standard 8x8 chess board with alternating dark green and off-white (or cream) squares. The board is oriented for White's perspective, with the files 'a' through 'h' labeled along the bottom edge (from left to right) and ranks '1' through '8' labeled along the left edge (from bottom to top). The "US CHESS FEDERATION" logo is visible at the top center of the board.

    **Detailed Breakdown of Pieces:**
    *   A **white queen** is positioned upright on square **e2** (a light-colored square).
    *   A **white pawn** is positioned upright on square **g6** (a dark green square).
    *   A **black pawn** is positioned upright on square **a3** (a dark green square).
    *   A **black pawn** is positioned upright on square **e4** (a dark green square).
    *   A **black rook** is lying on its side on square **g2** (a dark green square).

    **Other Details:**
    *   The lighting appears even, with no harsh shadows.
    *   The wooden surface beneath the board has a visible grain.
    *   The board itself appears to be a roll-up or flexible mat type.
    """
    
    # API endpoint
    url = "http://localhost:8001/recognize_chess_position_with_cursor_description"
    
    # Find a test image to use (we'll use any available image file)
    test_image_path = None
    for filename in ["test_image.jpg", "sample.jpeg", "IMG_4540.jpeg", "IMG_4545.jpg"]:
        if os.path.exists(filename):
            test_image_path = filename
            break
    
    if not test_image_path:
        print("❌ No test image found. Creating a dummy image...")
        # Create a simple test image
        import numpy as np
        import cv2
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        cv2.imwrite("test_dummy.jpg", test_img)
        test_image_path = "test_dummy.jpg"
    
    print(f"Using test image: {test_image_path}")
    
    # Prepare the form data
    files = {
        'image': (test_image_path, open(test_image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'cursor_description': cursor_description,
        'color': 'white'
    }
    
    print("Testing API with new image...")
    print(f"Cursor description length: {len(cursor_description)} characters")
    print(f"API endpoint: {url}")
    print(f"Test image: {test_image_path}")
    print("-" * 50)
    
    try:
        # Make the API call
        start_time = time.time()
        response = requests.post(url, files=files, data=data, timeout=30)
        end_time = time.time()
        
        print(f"Response status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print("-" * 50)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response:")
            print(f"FEN: {result.get('fen', 'N/A')}")
            print(f"Pieces Found: {result.get('pieces_found', 'N/A')}")
            print()
            
            print("🎯 2D Board Mapping:")
            board_2d = result.get('board_2d', [])
            if board_2d:
                print("   a b c d e f g h")
                print("  ─────────────────")
                for i, row in enumerate(board_2d):
                    rank = 8 - i
                    row_str = " ".join(row)
                    print(f"{rank} │{row_str}│")
                print("  ─────────────────")
                print("   a b c d e f g h")
            else:
                print("No 2D board mapping available")
            print()
            
            print("📊 Response Summary:")
            print(f"Response size: {len(json.dumps(result))} characters")
            print(f"Response keys: {list(result.keys())}")
            
            # Verify the expected pieces
            expected_pieces = {
                'e2': 'Q',  # White Queen
                'g6': 'P',  # White Pawn
                'a3': 'p',  # Black Pawn
                'e4': 'p',  # Black Pawn
                'g2': 'r'   # Black Rook
            }
            
            print("\n🔍 Piece Verification:")
            for square, expected_piece in expected_pieces.items():
                file = ord(square[0]) - ord('a')  # a=0, b=1, etc.
                rank = int(square[1]) - 1  # 1=0, 2=1, etc.
                board_rank = 7 - rank  # Convert to board array index
                actual_piece = board_2d[board_rank][file] if board_2d else '?'
                status = "✅" if actual_piece == expected_piece else "❌"
                print(f"  {status} {square}: expected {expected_piece}, got {actual_piece}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on port 8001")
        print("Run: python main.py")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: API request took too long")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Clean up dummy image if created
        if test_image_path == "test_dummy.jpg" and os.path.exists("test_dummy.jpg"):
            os.remove("test_dummy.jpg")

if __name__ == "__main__":
    test_api_with_new_image() 