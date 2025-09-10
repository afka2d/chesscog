#!/usr/bin/env python3
"""
Test the API flow to see where the pieces array gets corrupted.
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

def test_api_flow():
    """Test the API flow to see where pieces array gets corrupted."""
    
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
    
    logger.info(f"Testing API flow with image: {Path(image_path).name}")
    logger.info(f"Expected FEN: {expected_fen}")
    logger.info(f"Corners: {corners}")
    
    # Create recognizer
    try:
        recognizer = CustomChessRecognizer(Path("models"))
        logger.info("✅ Recognizer created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create recognizer: {e}")
        return False
    
    # Test the full API flow
    try:
        logger.info("Testing full API flow...")
        
        # Step 1: Classify occupancy
        logger.info("Step 1: Classifying occupancy...")
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners)
        logger.info(f"Occupancy shape: {occupancy.shape}, dtype: {occupancy.dtype}")
        
        # Step 2: Classify pieces
        logger.info("Step 2: Classifying pieces...")
        pieces = recognizer._classify_pieces(img, chess.WHITE, corners, occupancy)
        logger.info(f"Pieces shape: {pieces.shape}, dtype: {pieces.dtype}")
        
        # Check pieces before visualization
        logger.info("Pieces before visualization:")
        for i in range(8):
            for j in range(8):
                if pieces[i, j] is not None:
                    piece = pieces[i, j]
                    logger.info(f"  [{i},{j}]: {piece} (type: {type(piece)})")
                    if hasattr(piece, 'symbol'):
                        logger.info(f"    symbol(): {piece.symbol()}")
                    else:
                        logger.error(f"    No symbol() method! Type: {type(piece)}")
        
        # Step 3: Test visualization (this is where the error occurs)
        logger.info("Step 3: Testing visualization...")
        try:
            # Warp the board for visualization
            from chesscog.piece_classifier.create_dataset import warp_chessboard_image
            warped_board = warp_chessboard_image(img, corners)
            
            # Try to visualize piece map
            piece_map = recognizer._visualize_piece_map(warped_board, pieces, occupancy, chess.WHITE)
            logger.info("✅ Visualization successful!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            logger.error(f"Error type: {type(e)}")
            
            # Check pieces again after the error
            logger.info("Pieces after visualization error:")
            for i in range(8):
                for j in range(8):
                    if pieces[i, j] is not None:
                        piece = pieces[i, j]
                        logger.info(f"  [{i},{j}]: {piece} (type: {type(piece)})")
                        if hasattr(piece, 'symbol'):
                            logger.info(f"    symbol(): {piece.symbol()}")
                        else:
                            logger.error(f"    No symbol() method! Type: {type(piece)}")
            
            return False
        
    except Exception as e:
        logger.error(f"❌ API flow failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_flow()
    if success:
        logger.info("✅ API flow test passed!")
    else:
        logger.error("❌ API flow test failed!")
    exit(0 if success else 1)
