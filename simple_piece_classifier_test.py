#!/usr/bin/env python3
"""
Test piece classifier that always returns kings to verify API changes are reflected.
"""

import numpy as np
import chess
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class SimplePieceClassifierTest:
    """Test piece classifier that always returns kings for verification."""
    
    def __init__(self, models_dir: Path):
        """Initialize the test piece classifier."""
        self.models_dir = models_dir
        logger.info("🧪 TEST MODE: Piece classifier will return all KINGS for verification")
    
    def classify_pieces(self, img_array: np.ndarray, corners: np.ndarray, 
                       occupancy: List[bool], turn: chess.Color) -> List[Optional[chess.Piece]]:
        """
        Classify pieces - TEST VERSION that always returns kings.
        
        Args:
            img_array: Input image as numpy array
            corners: Board corner coordinates
            occupancy: List of 64 booleans indicating occupied squares
            turn: Current player's turn
            
        Returns:
            List of 64 chess.Piece objects or None
        """
        logger.info("🧪 TEST MODE: Returning all KINGS for verification")
        
        pieces = []
        for i in range(64):
            if bool(occupancy[i]):  # Only classify occupied squares
                # Always return a king (alternating colors for variety)
                if i % 2 == 0:
                    piece = chess.Piece(chess.KING, chess.WHITE)
                else:
                    piece = chess.Piece(chess.KING, chess.BLACK)
                pieces.append(piece)
            else:
                pieces.append(None)
        
        # Count pieces
        piece_count = sum(1 for p in pieces if p is not None)
        logger.info(f"🧪 TEST MODE: Generated {piece_count} KINGS out of {sum(occupancy)} occupied squares")
        
        return pieces
