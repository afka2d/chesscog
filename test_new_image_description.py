#!/usr/bin/env python3
"""
Test script for the new image description with different piece positions.
This script tests the updated parser with the new Cursor description format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import parse_cursor_description_to_board
import chess

def test_new_image_description():
    """Test the enhanced parser with the new Cursor description format."""
    
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
    
    print("Testing enhanced Cursor description parser with new image...")
    print("=" * 60)
    print("Input description:")
    print(cursor_description)
    print("=" * 60)
    
    # Parse the description
    board = parse_cursor_description_to_board(cursor_description)
    
    # Display results
    print("\nParsing Results:")
    print("-" * 30)
    print(f"FEN: {board.fen()}")
    print(f"Legal position: {board.is_valid()}")
    print(f"Number of pieces: {len([piece for piece in board.piece_map().values()])}")
    
    print("\nBoard visualization:")
    print("-" * 30)
    print(board)
    
    # Create and display 2D board mapping
    print("\n2D Board Mapping:")
    print("-" * 30)
    board_2d = []
    for rank in range(8):
        row = []
        for file in range(8):
            square = chess.square(file, 7 - rank)
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                row.append(piece_symbol)
            else:
                row.append('.')
        board_2d.append(row)
    
    # Print with rank numbers and file letters
    print("   a b c d e f g h")
    print("  ---------------")
    for i, row in enumerate(board_2d):
        print(f"{8-i} |{' '.join(row)}|")
    print("  ---------------")
    print("   a b c d e f g h")
    
    # List all pieces found
    print("\nPieces found:")
    print("-" * 30)
    piece_map = board.piece_map()
    if piece_map:
        for square, piece in piece_map.items():
            square_name = chess.square_name(square)
            piece_name = piece.symbol()
            color = "White" if piece.color == chess.WHITE else "Black"
            print(f"{color} {piece_name} on {square_name}")
    else:
        print("No pieces found!")
    
    print("\nExpected pieces from description:")
    print("-" * 30)
    expected_pieces = [
        ("White", "Q", "e2"),
        ("White", "P", "g6"), 
        ("Black", "P", "a3"),
        ("Black", "P", "e4"),
        ("Black", "R", "g2")
    ]
    
    for color, piece, square in expected_pieces:
        print(f"{color} {piece} on {square}")
    
    # Verify results
    print("\nVerification:")
    print("-" * 30)
    found_pieces = set()
    for square, piece in piece_map.items():
        square_name = chess.square_name(square)
        piece_symbol = piece.symbol()
        color = "White" if piece.color == chess.WHITE else "Black"
        found_pieces.add((color, piece_symbol, square_name))
    
    expected_pieces_set = set(expected_pieces)
    
    if found_pieces == expected_pieces_set:
        print("✅ SUCCESS: All expected pieces were found!")
    else:
        print("❌ FAILURE: Some pieces were missing or incorrect")
        print(f"Expected: {expected_pieces_set}")
        print(f"Found: {found_pieces}")
        missing = expected_pieces_set - found_pieces
        extra = found_pieces - expected_pieces_set
        if missing:
            print(f"Missing: {missing}")
        if extra:
            print(f"Extra: {extra}")

if __name__ == "__main__":
    test_new_image_description() 