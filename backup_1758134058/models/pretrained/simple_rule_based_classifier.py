#!/usr/bin/env python3
"""
Simple rule-based chess piece classifier.
Uses position-based heuristics instead of machine learning.
"""

import numpy as np
import chess

class SimpleChessPieceClassifier:
    def __init__(self):
        self.class_names = [
            'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
            'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
        ]
    
    def classify_pieces(self, occupancy, turn):
        """Classify pieces using simple rules based on position and occupancy."""
        pieces = np.full((8, 8), None, dtype=object)
        
        for rank in range(8):
            for file in range(8):
                if occupancy[rank, file]:  # If square is occupied
                    piece = self._get_piece_by_position(rank, file, turn)
                    pieces[rank, file] = piece
        
        return pieces
    
    def _get_piece_by_position(self, rank, file, turn):
        """Get piece based on position using chess rules."""
        # Convert to chess square
        square = chess.square(file, 7 - rank)
        
        # Simple rules based on typical piece positions
        if rank == 0 or rank == 7:  # Back rank
            if file == 0 or file == 7:  # Corners
                piece_type = chess.ROOK
            elif file == 1 or file == 6:  # Knight positions
                piece_type = chess.KNIGHT
            elif file == 2 or file == 5:  # Bishop positions
                piece_type = chess.BISHOP
            elif file == 3:  # Queen position
                piece_type = chess.QUEEN
            elif file == 4:  # King position
                piece_type = chess.KING
            else:
                piece_type = chess.PAWN
        else:  # Other ranks
            piece_type = chess.PAWN
        
        # Determine color based on turn and position
        if (turn == chess.WHITE and rank < 4) or (turn == chess.BLACK and rank >= 4):
            color = chess.WHITE
        else:
            color = chess.BLACK
        
        return chess.Piece(piece_type, color)
