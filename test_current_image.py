#!/usr/bin/env python3
"""
Test script to call the API with the current attached image.
This tests the /recognize_chess_position_with_cursor_description endpoint.
"""

import requests
import json
import time
import os

def test_api_with_current_image():
    """Test the API with the current attached image."""
    
    # The Cursor description from the current image
    cursor_description = """
    This image displays a chess board with several pieces, viewed from a slightly elevated angle, resting on a light-colored wooden surface.

    **High-Level Description:**
    The image shows a standard 8x8 chess board with alternating dark green and off-white (or cream) squares. The board is oriented for White's perspective, with the files 'a' through 'h' labeled along the bottom edge (from left to right) and ranks '1' through '8' labeled along the left edge (from bottom to top). There are five chess pieces on the board: two white pieces and three black pieces. One black piece is lying on its side. The "US CHESS FEDERATION" logo is visible at the bottom center of the board, and a stylized chess piece logo is in the bottom right corner (h1 square).

    **Detailed Breakdown of the Chess Board and Pieces:**

    *   **Chess Board:**
        *   **Dimensions:** Standard 8x8 grid.
        *   **Colors:** Dark green and off-white/cream squares.
        *   **Orientation:** Files 'a' through 'h' are labeled along the bottom edge, and ranks '1' through '8' are labeled along the left edge, indicating a standard setup from White's perspective.
        *   **Branding:** The "US CHESS FEDERATION" logo is printed on the border below the 'd' and 'e' files.
        *   **Corner Logo:** A decorative logo, resembling a stylized chess piece (possibly a knight or a crown), is present on the border near the h1 square.

    *   **Chess Pieces:**
        *   **White Pieces:**
            *   A **white queen** is positioned upright on square **e2** (a light-colored square).
            *   A **white pawn** is positioned upright on square **g6** (a dark green square).
        *   **Black Pieces:**
            *   A **black pawn** is positioned upright on square **a3** (a dark green square).
            *   A **black pawn** is positioned upright on square **e4** (a dark green square).
            *   A **black rook** is lying on its side on square **g2** (a light-colored square). Its base is facing towards the 'h' file, and its top is pointing towards the 'f' file.

    **Overall Scene:**
    The board appears to be a roll-up or flexible mat, given its slight waviness. The wooden surface beneath it has a light, natural grain. The lighting is even, with no harsh shadows, suggesting an indoor setting.
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
    
    print("Testing API with current image...")
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
    test_api_with_current_image() 