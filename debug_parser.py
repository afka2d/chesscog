#!/usr/bin/env python3
"""
Debug script to see what the parser is receiving and why it's not finding pieces.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import parse_cursor_description_to_board
import chess

def debug_parser():
    """Debug the parser with the new image description."""
    
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
    
    print("Debugging parser with new image description...")
    print("=" * 60)
    
    # Let's manually process the description to see what's happening
    lines = cursor_description.split('\n')
    print("Processing lines:")
    print("-" * 30)
    
    import re
    
    # Enhanced patterns to handle Cursor's bullet-pointed format
    patterns = [
        # Original patterns
        r'(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+on\s+([a-h][1-8])',
        r'(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+positioned\s+on\s+([a-h][1-8])',
        r'(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+at\s+([a-h][1-8])',
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+positioned\s+on\s+([a-h][1-8])',
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+on\s+([a-h][1-8])',
        # New patterns for Cursor's bullet-pointed format
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+positioned\s+on\s+square\s+([a-h][1-8])',
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+positioned\s+on\s+([a-h][1-8])',
        r'(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+positioned\s+on\s+square\s+([a-h][1-8])',
        r'(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+positioned\s+on\s+([a-h][1-8])',
        # Handle "lying on its side" or other variations
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+lying\s+on\s+its\s+side\s+on\s+square\s+([a-h][1-8])',
        r'a\s+(\w+)\s+(pawn|rook|knight|bishop|queen|king)\s+is\s+lying\s+on\s+its\s+side\s+on\s+([a-h][1-8])',
    ]
    
    pieces_found = []
    found_squares = set()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Remove bullet points and extra whitespace
        line = re.sub(r'^[-•*]\s*', '', line)
        line = re.sub(r'^\*\s*', '', line)
        line = line.strip()
        
        if not line:
            continue
            
        line_lower = line.lower()
        
        print(f"Line {i+1}: '{line}'")
        print(f"  Lower: '{line_lower}'")
        
        # Try all patterns on this line
        for j, pattern in enumerate(patterns):
            matches = re.findall(pattern, line_lower)
            if matches:
                print(f"  Pattern {j+1} matched: {matches}")
                for match in matches:
                    color, piece_type, square = match
                    piece_key = f"{color} {piece_type}"
                    print(f"    Found: {piece_key} on {square}")
                    
                    # Check if this piece and square are valid
                    piece_map = {
                        'white pawn': chess.Piece(chess.PAWN, chess.WHITE),
                        'white rook': chess.Piece(chess.ROOK, chess.WHITE),
                        'white knight': chess.Piece(chess.KNIGHT, chess.WHITE),
                        'white bishop': chess.Piece(chess.BISHOP, chess.WHITE),
                        'white queen': chess.Piece(chess.QUEEN, chess.WHITE),
                        'white king': chess.Piece(chess.KING, chess.WHITE),
                        'black pawn': chess.Piece(chess.PAWN, chess.BLACK),
                        'black rook': chess.Piece(chess.ROOK, chess.BLACK),
                        'black knight': chess.Piece(chess.KNIGHT, chess.BLACK),
                        'black bishop': chess.Piece(chess.BISHOP, chess.BLACK),
                        'black queen': chess.Piece(chess.QUEEN, chess.BLACK),
                        'black king': chess.Piece(chess.KING, chess.BLACK),
                    }
                    
                    square_map = {
                        'a1': chess.A1, 'a2': chess.A2, 'a3': chess.A3, 'a4': chess.A4,
                        'a5': chess.A5, 'a6': chess.A6, 'a7': chess.A7, 'a8': chess.A8,
                        'b1': chess.B1, 'b2': chess.B2, 'b3': chess.B3, 'b4': chess.B4,
                        'b5': chess.B5, 'b6': chess.B6, 'b7': chess.B7, 'b8': chess.B8,
                        'c1': chess.C1, 'c2': chess.C2, 'c3': chess.C3, 'c4': chess.C4,
                        'c5': chess.C5, 'c6': chess.C6, 'c7': chess.C7, 'c8': chess.C8,
                        'd1': chess.D1, 'd2': chess.D2, 'd3': chess.D3, 'd4': chess.D4,
                        'd5': chess.D5, 'd6': chess.D6, 'd7': chess.D7, 'd8': chess.D8,
                        'e1': chess.E1, 'e2': chess.E2, 'e3': chess.E3, 'e4': chess.E4,
                        'e5': chess.E5, 'e6': chess.E6, 'e7': chess.E7, 'e8': chess.E8,
                        'f1': chess.F1, 'f2': chess.F2, 'f3': chess.F3, 'f4': chess.F4,
                        'f5': chess.F5, 'f6': chess.F6, 'f7': chess.F7, 'f8': chess.F8,
                        'g1': chess.G1, 'g2': chess.G2, 'g3': chess.G3, 'g4': chess.G4,
                        'g5': chess.G5, 'g6': chess.G6, 'g7': chess.G7, 'g8': chess.G8,
                        'h1': chess.H1, 'h2': chess.H2, 'h3': chess.H3, 'h4': chess.H4,
                        'h5': chess.H5, 'h6': chess.H6, 'h7': chess.H7, 'h8': chess.H8,
                    }
                    
                    if piece_key in piece_map and square in square_map:
                        if square not in found_squares:
                            pieces_found.append((piece_map[piece_key], square_map[square]))
                            found_squares.add(square)
                            print(f"    ✅ Added: {piece_key} on {square}")
                        else:
                            print(f"    ⚠️ Skipping duplicate on {square}")
                    else:
                        print(f"    ❌ Invalid: piece_key='{piece_key}' in piece_map: {piece_key in piece_map}, square='{square}' in square_map: {square in square_map}")
    
    print(f"\nTotal pieces found: {len(pieces_found)}")
    print(f"Pieces: {pieces_found}")
    
    # Now test the actual parser
    print("\n" + "="*60)
    print("Testing actual parser function:")
    print("="*60)
    
    board = parse_cursor_description_to_board(cursor_description)
    print(f"Parser result - FEN: {board.fen()}")
    print(f"Parser result - Pieces: {len([piece for piece in board.piece_map().values()])}")

if __name__ == "__main__":
    debug_parser() 