#!/usr/bin/env python3
"""
Debug script to see what the parser is receiving and why it's not finding pieces.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import parse_cursor_description_to_board
import chess

def debug_current_parser():
    """Debug the parser with the current image description."""
    
    # The current Cursor description from the image
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
    
    print("🔍 Debugging Current Parser")
    print("=" * 50)
    print(f"Description length: {len(cursor_description)} characters")
    print()
    
    print("📝 Raw Description (first 500 chars):")
    print(cursor_description[:500])
    print("...")
    print()
    
    # Test the parser
    print("🎯 Testing Parser...")
    board = parse_cursor_description_to_board(cursor_description)
    
    # Check what pieces were found
    piece_map = board.piece_map()
    print(f"Pieces found: {len(piece_map)}")
    
    if piece_map:
        print("✅ Found pieces:")
        for square, piece in piece_map.items():
            square_name = chess.square_name(square)
            piece_symbol = piece.symbol()
            print(f"  - {piece_symbol} on {square_name}")
    else:
        print("❌ No pieces found!")
        print()
        print("🔍 Let's analyze why...")
        print()
        
        # Let's look at the lines that should contain pieces
        lines = cursor_description.split('\n')
        print("📋 Lines that might contain pieces:")
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['queen', 'pawn', 'rook', 'bishop', 'knight', 'king']):
                print(f"Line {i+1}: {line.strip()}")
        
        print()
        print("🔍 The issue might be with the formatting. Let's check the regex patterns...")
        
        # Let's manually test some patterns
        import re
        
        # Test the patterns from the parser
        patterns = [
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+on\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+positioned\s+on\s+square\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+is\s+positioned\s+on\s+square\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+is\s+positioned\s+upright\s+on\s+square\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+positioned\s+upright\s+on\s+square\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+lying\s+on\s+its\s+side\s+on\s+square\s+([a-h][1-8])',
            r'(\w+)\s+(queen|pawn|rook|bishop|knight|king)\s+is\s+lying\s+on\s+its\s+side\s+on\s+square\s+([a-h][1-8])'
        ]
        
        print("🧪 Testing regex patterns on key lines:")
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['queen', 'pawn', 'rook', 'bishop', 'knight', 'king']):
                print(f"\nLine {i+1}: {line.strip()}")
                for j, pattern in enumerate(patterns):
                    matches = re.findall(pattern, line_lower)
                    if matches:
                        print(f"  Pattern {j+1} matched: {matches}")
    
    print()
    print("🎯 Final Board State:")
    print(board)
    print(f"FEN: {board.fen()}")

if __name__ == "__main__":
    debug_current_parser() 