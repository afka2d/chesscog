#!/usr/bin/env python3
"""
Debug script to test piece classification directly.
"""

import numpy as np
import chess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_piece_conversion():
    """Test the piece conversion logic."""
    
    # Test piece names from the parent class
    piece_names = [
        'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king', 'white_pawn',
        'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king', 'black_pawn'
    ]
    
    logger.info("Testing piece name conversion...")
    
    for piece_name in piece_names:
        if piece_name.startswith('white_'):
            color = chess.WHITE
            piece_type = piece_name[6:]  # Remove 'white_' prefix
        else:
            color = chess.BLACK
            piece_type = piece_name[6:]  # Remove 'black_' prefix
        
        piece_map = {
            'pawn': chess.PAWN,
            'rook': chess.ROOK,
            'knight': chess.KNIGHT,
            'bishop': chess.BISHOP,
            'queen': chess.QUEEN,
            'king': chess.KING
        }
        
        if piece_type in piece_map:
            piece_obj = chess.Piece(piece_map[piece_type], color)
            logger.info(f"{piece_name} -> {piece_obj} (type: {type(piece_obj)})")
        else:
            logger.error(f"Unknown piece type: {piece_type}")

def test_array_creation():
    """Test creating a 2D pieces array."""
    
    logger.info("Testing 2D array creation...")
    
    # Create a 2D array
    pieces_2d = np.full((8, 8), None, dtype=object)
    logger.info(f"Array shape: {pieces_2d.shape}")
    logger.info(f"Array dtype: {pieces_2d.dtype}")
    
    # Add some pieces
    pieces_2d[0, 0] = chess.Piece(chess.ROOK, chess.WHITE)
    pieces_2d[7, 7] = chess.Piece(chess.ROOK, chess.BLACK)
    
    logger.info("Pieces added:")
    for i in range(8):
        for j in range(8):
            if pieces_2d[i, j] is not None:
                piece = pieces_2d[i, j]
                logger.info(f"  [{i},{j}]: {piece} (type: {type(piece)})")
                logger.info(f"    symbol(): {piece.symbol()}")
                logger.info(f"    piece_type: {piece.piece_type}")
                logger.info(f"    color: {piece.color}")

if __name__ == "__main__":
    test_piece_conversion()
    print()
    test_array_creation()
