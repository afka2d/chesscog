#!/usr/bin/env python3
"""
Clean main.py with improved occupancy detection and piece classification.
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
from simple_piece_classifier import SimplePieceClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
piece_classifier = None

# FastAPI app
app = FastAPI(title="Chess Position Scanner API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the chess recognizer on startup."""
    global piece_classifier
    
    logger.info("Starting up Chess Position Scanner API...")
    
    # Load configuration
    logger.info("Loading configuration...")
    config_path = Path("models/recognition.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    logger.info("Configuration loaded successfully")
    
    # Initialize piece classifier
    logger.info("Initializing piece classifier...")
    piece_classifier = SimplePieceClassifier(Path("models"))
    logger.info("Piece classifier initialized successfully")
    
    logger.info("Startup completed successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "piece_classifier_loaded": piece_classifier is not None
    }

def simple_occupancy_detection(img_array, corners):
    """Simple, stable occupancy detection."""
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
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Divide into 8x8 grid
        square_size = board_size // 8
        occupancy = []
        
        for row in range(8):
            for col in range(8):
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size
                
                # Extract square from both original and thresholded images
                square_img = gray[y1:y2, x1:x2]
                square_thresh = thresh[y1:y2, x1:x2]
                
                # Calculate statistics
                mean_val = np.mean(square_img)
                std_dev = np.std(square_img)
                
                # Count white pixels in thresholded image (edges)
                white_pixels = np.sum(square_thresh == 255)
                total_pixels = square_thresh.size
                edge_ratio = white_pixels / total_pixels
                
                # More sophisticated occupancy detection
                is_occupied = (
                    std_dev > 20 or  # High variation
                    abs(mean_val - 128) > 25 or  # Different from background
                    edge_ratio > 0.1  # Significant edge content
                )
                occupancy.append(is_occupied)
        
        occupied_count = sum(occupancy)
        logger.info(f"Simple occupancy detection found {occupied_count} occupied squares out of 64")
        return occupancy
        
    except Exception as e:
        logger.error(f"Simple occupancy detection failed: {e}")
        # Ultimate fallback: assume all squares are occupied
        logger.warning("Using ultimate fallback: assuming all squares occupied")
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
        
        # Use simple, stable occupancy detection
        logger.info("Detecting occupancy using simple method...")
        occupancy = simple_occupancy_detection(img_array, corners_list)
        
        # Count occupied squares
        occupied_count = sum(occupancy)
        logger.info(f"Detected {occupied_count} occupied squares out of 64")
        
        # Classify pieces on occupied squares
        logger.info("Classifying pieces with custom classifier...")
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
        
        return {
            "fen": fen,
            "pieces": pieces,
            "occupancy": occupancy_response,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in recognize_chess_position_with_corners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)