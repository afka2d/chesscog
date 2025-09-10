#!/usr/bin/env python3
"""
Test main.py that uses the test piece classifier to verify API changes are reflected.
"""

import os
import logging
import numpy as np
import chess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import cv2
import json
from simple_piece_classifier_test import SimplePieceClassifierTest
from chesscog.recognition.recognition import ChessRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
piece_classifier = None
occupancy_recognizer = None

# FastAPI app
app = FastAPI(title="Chess Position Scanner API - TEST MODE", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the chess recognizer on startup."""
    global piece_classifier, occupancy_recognizer
    
    logger.info("🧪 Starting up Chess Position Scanner API - TEST MODE")
    logger.info("🧪 TEST MODE: All pieces will be classified as KINGS for verification")
    
    # Initialize test piece classifier
    logger.info("Initializing TEST piece classifier...")
    piece_classifier = SimplePieceClassifierTest(Path("models"))
    logger.info("TEST piece classifier initialized successfully")
    
    # Initialize occupancy recognizer (keep your working one)
    logger.info("Initializing occupancy recognizer...")
    occupancy_recognizer = ChessRecognizer(Path("models"))
    logger.info("Occupancy recognizer initialized successfully")
    
    logger.info("🧪 TEST MODE: Startup completed successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy - TEST MODE", 
        "piece_classifier_loaded": piece_classifier is not None,
        "occupancy_recognizer_loaded": occupancy_recognizer is not None,
        "test_mode": "All pieces will be classified as KINGS"
    }

def get_occupancy_with_fallback(img_array, corners, turn):
    """Get occupancy using your working ChessCog method with proper error handling."""
    try:
        # Use your working ChessCog occupancy detection
        logger.info("Using ChessCog occupancy detection...")
        board, detected_corners = occupancy_recognizer.predict(img_array, turn)
        
        # Extract occupancy from the board
        occupancy = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            # Handle the array ambiguity error by checking if piece is not None
            occupancy.append(piece is not None)
        
        occupied_count = sum(occupancy)
        logger.info(f"ChessCog detected {occupied_count} occupied squares out of 64")
        return occupancy
        
    except Exception as e:
        logger.warning(f"ChessCog occupancy detection failed: {e}")
        logger.info("Falling back to simple occupancy detection...")
        
        # Fallback to simple occupancy detection
        try:
            # Convert corners to numpy array
            corners = np.array(corners, dtype=np.float32)
            
            # Get perspective transformation
            board_size = 400
            dst_points = np.array([
                [0, 0],
                [board_size, 0],
                [board_size, board_size],
                [0, board_size]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(corners, dst_points)
            warped = cv2.warpPerspective(img_array, M, (board_size, board_size))
            
            # Convert to grayscale
            gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            
            # Divide into 8x8 grid
            square_size = board_size // 8
            occupancy = []
            
            for row in range(8):
                for col in range(8):
                    y1 = row * square_size
                    y2 = (row + 1) * square_size
                    x1 = col * square_size
                    x2 = (col + 1) * square_size
                    
                    square = gray[y1:y2, x1:x2]
                    
                    # More conservative thresholds for better occupancy detection
                    std_dev = np.std(square)
                    mean_val = np.mean(square)
                    
                    # Only mark as occupied if there's significant variation
                    is_occupied = std_dev > 30 and abs(mean_val - 128) > 25
                    occupancy.append(is_occupied)
            
            occupied_count = sum(occupancy)
            logger.info(f"Simple detection found {occupied_count} occupied squares out of 64")
            return occupancy
            
        except Exception as e2:
            logger.warning(f"Simple occupancy detection also failed: {e2}")
            # Final fallback to all squares occupied
            return [True] * 64

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    color: str = Form("white")
):
    """Recognize chess position from image with corner coordinates."""
    try:
        # Parse corners
        corners_list = json.loads(corners)
        corners_array = np.array(corners_list, dtype=np.float32)
        
        # Read and process image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        # Determine turn
        turn = chess.WHITE if color.lower() == "white" else chess.BLACK
        
        # Use your working occupancy detection
        logger.info("Detecting occupancy with working method...")
        occupancy = get_occupancy_with_fallback(img_array, corners_list, turn)
        
        # Count occupied squares
        occupied_count = sum(occupancy)
        logger.info(f"Detected {occupied_count} occupied squares out of 64")
        
        # Classify pieces on occupied squares - TEST MODE
        logger.info("🧪 TEST MODE: Classifying pieces as KINGS...")
        pieces_1d = piece_classifier.classify_pieces(img_array, corners_array, occupancy, turn)
        
        # Convert 1D result to 2D for consistency
        pieces_2d = np.full((8, 8), None, dtype=object)
        for i, piece in enumerate(pieces_1d):
            rank, file = i // 8, i % 8
            if piece is not None:
                pieces_2d[rank, file] = piece
        
        # Convert pieces_2d to 1D list for API response
        pieces = []
        for rank in range(8):
            for file in range(8):
                piece = pieces_2d[rank, file]
                if piece is not None:
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    pieces.append(piece_name)
                else:
                    pieces.append(None)
        
        # Create a new board with the classified pieces
        new_board = chess.Board()
        new_board.clear()
        for rank in range(8):
            for file in range(8):
                piece = pieces_2d[rank, file]
                if piece is not None:
                    square = chess.square(file, 7-rank)  # Convert to chess square
                    new_board.set_piece_at(square, piece)
        
        # Convert board to FEN
        fen = new_board.fen()
        
        # Create occupancy array for response (based on actual pieces detected)
        occupancy_response = []
        for rank in range(8):
            for file in range(8):
                piece = pieces_2d[rank, file]
                occupancy_response.append(piece is not None)
        
        # Count kings for verification
        king_count = sum(1 for p in pieces if p and 'king' in p)
        logger.info(f"🧪 TEST MODE: Generated {king_count} KINGS in response")
        
        return {
            "fen": fen,
            "pieces": pieces,
            "occupancy": occupancy_response,
            "success": True,
            "test_mode": "All pieces classified as KINGS for verification"
        }
        
    except Exception as e:
        logger.error(f"Error in recognize_chess_position_with_corners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
