#!/usr/bin/env python3
"""
Decode FEN string to show what pieces should be on the board.
"""

import chess

def decode_fen(fen):
    """Decode FEN and show board position."""
    board = chess.Board(fen)
    
    print(f"FEN: {fen}")
    print("\nBoard position:")
    print(board)
    
    print("\nPiece locations:")
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            piece_name = piece.symbol()
            color = "white" if piece.color else "black"
            piece_type = {
                'p': 'pawn', 'r': 'rook', 'n': 'knight',
                'b': 'bishop', 'q': 'queen', 'k': 'king'
            }[piece_name.lower()]
            print(f"{square_name}: {color} {piece_type} ({piece_name})")

if __name__ == "__main__":
    fen = "8/3k4/2n1q3/1n1p1p2/4P3/2N2P2/PPP5/1N1Q4 w - - 0 1"
    decode_fen(fen)