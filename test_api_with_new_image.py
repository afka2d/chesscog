#!/usr/bin/env python3
"""
Test script to call the API endpoint with the new image description.
This tests the /recognize_chess_position_with_cursor_description endpoint.
"""

import requests
import json
import time

def test_api_with_new_image():
    """Test the API with the new image description."""
    
    # The new Cursor description from the image
    cursor_description = """
    This image displays a chess board with several pieces, viewed from a slightly elevated angle, resting on a light-colored wooden surface.

    Here's a detailed breakdown of the image:

    **Chess Board:**
    *   It is a standard 8x8 grid with alternating dark green and off-white (or cream) squares.
    *   The board is oriented for White's perspective, with the files 'a' through 'h' labeled along the bottom edge (from left to right) and ranks '1' through '8' labeled along the left edge (from bottom to top).
    *   The bottom edge of the board, near the 'c' and 'd' files, features the "US CHESS FEDERATION" logo.
    *   The bottom right corner (h1 square) has a small, stylized chess piece icon.

    **Chess Pieces:**
    There are five chess pieces visible on the board:

    *   **White Pieces:**
        *   A white queen is positioned on square e2 (a light-colored square). It is standing upright.
        *   A white pawn is positioned on square g6 (a dark green square). It is standing upright.

    *   **Black Pieces:**
        *   A black pawn is positioned on square a3 (a dark green square). It is standing upright.
        *   A black pawn is positioned on square e4 (a dark green square). It is standing upright.
        *   A black rook is positioned on square g2 (a light-colored square). It is lying on its side, with its base facing towards the 'h' file and its top towards the 'f' file.
    """
    
    print("Testing API with new image description...")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8001/recognize_chess_position_with_cursor_description"
    
    # Create a dummy image file (since we're only testing the description parsing)
    # We'll use a simple test image
    test_image_path = "sample.jpeg"  # Use an existing image file
    
    try:
        # Prepare the request
        with open(test_image_path, 'rb') as image_file:
            files = {'image': ('test_image.jpg', image_file, 'image/jpeg')}
            data = {
                'cursor_description': cursor_description,
                'color': 'white'
            }
            
            print(f"Sending request to: {url}")
            print(f"Image file: {test_image_path}")
            print(f"Description length: {len(cursor_description)} characters")
            
            # Make the request
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            end_time = time.time()
            
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Time: {end_time - start_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n" + "="*60)
                print("API RESPONSE:")
                print("="*60)
                
                # Display key results
                print(f"FEN: {result.get('fen', 'N/A')}")
                print(f"Legal Position: {result.get('legal_position', 'N/A')}")
                print(f"Pieces Found: {result.get('pieces_found', 'N/A')}")
                print(f"Method: {result.get('method', 'N/A')}")
                
                # Display position description
                print(f"\nPosition Description:")
                print(f"{result.get('position_description', 'N/A')}")
                
                # Display 2D board mapping
                print(f"\n2D Board Mapping:")
                board_2d = result.get('board_2d', [])
                if board_2d:
                    print("   a b c d e f g h")
                    print("  ---------------")
                    for i, row in enumerate(board_2d):
                        print(f"{8-i} |{' '.join(row)}|")
                    print("  ---------------")
                    print("   a b c d e f g h")
                else:
                    print("No 2D board mapping found!")
                
                # Display ASCII board
                print(f"\nASCII Board:")
                print(result.get('ascii', 'N/A'))
                
                # Display Lichess URL
                print(f"\nLichess URL:")
                print(result.get('lichess_url', 'N/A'))
                
                # Display debug info
                print(f"\nDebug Info:")
                debug_info = result.get('debug_info', {})
                for key, value in debug_info.items():
                    print(f"  {key}: {value}")
                
                # Verify the pieces
                print(f"\nVerification:")
                print("-" * 30)
                expected_pieces = [
                    ("White", "Q", "e2"),
                    ("White", "P", "g6"), 
                    ("Black", "P", "a3"),
                    ("Black", "P", "e4"),
                    ("Black", "R", "g2")
                ]
                
                print("Expected pieces:")
                for color, piece, square in expected_pieces:
                    print(f"  {color} {piece} on {square}")
                
                # Check if all pieces are in the 2D board
                found_pieces = []
                for rank in range(8):
                    for file in range(8):
                        piece_symbol = board_2d[rank][file]
                        if piece_symbol != '.':
                            square_name = chr(ord('a') + file) + str(8 - rank)
                            color = "White" if piece_symbol.isupper() else "Black"
                            piece_name = piece_symbol.upper()
                            found_pieces.append((color, piece_name, square_name))
                
                print("\nFound pieces in 2D board:")
                for color, piece, square in found_pieces:
                    print(f"  {color} {piece} on {square}")
                
                # Check if all expected pieces are found
                expected_set = set(expected_pieces)
                found_set = set(found_pieces)
                
                if expected_set == found_set:
                    print("\n✅ SUCCESS: All expected pieces found in 2D board!")
                else:
                    print("\n❌ FAILURE: Some pieces missing or incorrect")
                    missing = expected_set - found_set
                    extra = found_set - expected_set
                    if missing:
                        print(f"Missing: {missing}")
                    if extra:
                        print(f"Extra: {extra}")
                
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print(f"Error: Image file {test_image_path} not found!")
        print("Please ensure you have a test image file available.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server!")
        print("Make sure the server is running on http://localhost:8001")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api_with_new_image() 