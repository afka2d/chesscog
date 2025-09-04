#!/usr/bin/env python3
"""
Verify the coordinate system and FEN interpretation for piece extraction.
"""

import cv2
import numpy as np
import chess

def verify_coordinates():
    """Verify the coordinate system and piece positions."""
    print("🔍 Verifying coordinate system and piece positions...")
    
    # FEN: 3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1
    fen = "3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1"
    
    # Parse FEN
    board = chess.Board(fen)
    
    print(f"📝 FEN: {fen}")
    print(f"📊 Board dimensions: 8x8")
    
    # Show the board representation
    print(f"\n🎯 Board representation (from White's perspective - bottom):")
    print("   a b c d e f g h")
    print("  ─────────────────")
    
    for rank in range(8):
        rank_str = f"{8-rank} "
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is not None:
                rank_str += piece.symbol() + " "
            else:
                rank_str += ". "
        print(f"  {rank_str}{8-rank}")
    
    print("  ─────────────────")
    print("   a b c d e f g h")
    
    # Now show from Black's perspective (top)
    print(f"\n🎯 Board representation (from Black's perspective - top):")
    print("   h g f e d c b a")
    print("  ─────────────────")
    
    for rank in range(8):
        rank_str = f"{rank+1} "
        for file in range(7, -1, -1):  # Reverse file order
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is not None:
                rank_str += piece.symbol() + " "
            else:
                rank_str += ". "
        print(f"  {rank_str}{rank+1}")
    
    print("  ─────────────────")
    print("   h g f e d c b a")
    
    # Check specific squares mentioned in the issue
    print(f"\n🔍 Checking specific squares:")
    
    # d8 should contain a black rook
    d8_square = chess.square(3, 0)  # file 3 (d), rank 0 (8)
    d8_piece = board.piece_at(d8_square)
    print(f"   d8 (file 3, rank 0): {d8_piece.symbol() if d8_piece else 'empty'} - {d8_piece.color if d8_piece else 'N/A'}")
    
    # f8 should contain a black rook
    f8_square = chess.square(5, 0)  # file 5 (f), rank 0 (8)
    f8_piece = board.piece_at(f8_square)
    print(f"   f8 (file 5, rank 0): {f8_piece.symbol() if f8_piece else 'empty'} - {f8_piece.color if d8_piece else 'N/A'}")
    
    # h7 should contain a black king
    h7_square = chess.square(7, 1)  # file 7 (h), rank 1 (7)
    h7_piece = board.piece_at(h7_square)
    print(f"   h7 (file 7, rank 1): {h7_piece.symbol() if h7_piece else 'empty'} - {h7_piece.color if h7_piece else 'N/A'}")
    
    # a4 should contain a white queen
    a4_square = chess.square(0, 4)  # file 0 (a), rank 4 (4)
    a4_piece = board.piece_at(a4_square)
    print(f"   a4 (file 0, rank 4): {a4_piece.symbol() if a4_piece else 'empty'} - {a4_piece.color if a4_piece else 'N/A'}")
    
    print(f"\n🔍 Coordinate system analysis:")
    print(f"   - Chess coordinates: a1=bottom-left, h8=top-right")
    print(f"   - Array coordinates: [0,0]=top-left, [7,7]=bottom-right")
    print(f"   - FEN reads from rank 8 (top) to rank 1 (bottom)")
    print(f"   - Each rank reads from a (left) to h (right)")
    
    # Show the mapping
    print(f"\n📐 Coordinate mapping:")
    print(f"   Chess square | File | Rank | Array [file, rank] | Piece")
    print(f"   ──────────────────────────────────────────────────────")
    
    test_squares = ['d8', 'f8', 'h7', 'a4', 'd5', 'e4']
    for square_name in test_squares:
        # Parse square name
        file = ord(square_name[0]) - ord('a')  # a=0, b=1, c=2, etc.
        rank = 8 - int(square_name[1])         # 8=0, 7=1, 6=2, etc.
        
        square = chess.square(file, rank)
        piece = board.piece_at(square)
        piece_symbol = piece.symbol() if piece else '.'
        
        print(f"   {square_name:>4}        | {file:>4} | {rank:>4} | [{file:>2}, {rank:>2}]         | {piece_symbol}")

if __name__ == "__main__":
    verify_coordinates()
