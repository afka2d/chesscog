#!/usr/bin/env python3
"""
Test piece classification directly without the API.
"""

import cv2
import numpy as np
import chess
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the custom recognizer
import sys
sys.path.append('.')
from main import CustomChessRecognizer

def test_direct_classification():
    """Test piece classification directly."""
    
    # Load a test image
    image_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    annotation_path = "grey_background_dataset/annotations/test/IMG_4763.json"
    
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return False
    
    if not Path(annotation_path).exists():
        logger.error(f"Annotation not found: {annotation_path}")
        return False
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    corners = np.array(annotation["corners"], dtype=np.float32)
    expected_fen = annotation["fen"]
    
    logger.info(f"Testing with image: {Path(image_path).name}")
    logger.info(f"Expected FEN: {expected_fen}")
    logger.info(f"Corners: {corners}")
    
    # Create recognizer
    try:
        recognizer = CustomChessRecognizer(Path("models"))
        logger.info("✅ Recognizer created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create recognizer: {e}")
        return False
    
    # Test occupancy classification
    try:
        logger.info("Testing occupancy classification...")
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners)
        logger.info(f"Occupancy shape: {occupancy.shape}")
        logger.info(f"Occupancy dtype: {occupancy.dtype}")
        logger.info(f"Occupied squares: {np.sum(occupancy)}")
    except Exception as e:
        logger.error(f"❌ Occupancy classification failed: {e}")
        return False
    
    # Test piece classification
    try:
        logger.info("Testing piece classification...")
        pieces = recognizer._classify_pieces(img, chess.WHITE, corners, occupancy)
        logger.info(f"Pieces shape: {pieces.shape}")
        logger.info(f"Pieces dtype: {pieces.dtype}")
        
        # Check what's in the pieces array
        for i in range(8):
            for j in range(8):
                if pieces[i, j] is not None:
                    piece = pieces[i, j]
                    logger.info(f"Piece at [{i},{j}]: {piece} (type: {type(piece)})")
                    if hasattr(piece, 'symbol'):
                        logger.info(f"  symbol(): {piece.symbol()}")
                    else:
                        logger.error(f"  No symbol() method! Type: {type(piece)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Piece classification failed: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_classification()
    if success:
        logger.info("✅ Direct classification test passed!")
    else:
        logger.error("❌ Direct classification test failed!")
    exit(0 if success else 1)
