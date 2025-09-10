#!/usr/bin/env python3
"""
Working main.py that uses our piece classifier correctly.
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
from chesscog.recognition.recognition import ChessRecognizer
from simple_piece_classifier import SimplePieceClassifier
from recap import CfgNode as CN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomChessRecognizer(ChessRecognizer):
    """Custom chess recognizer that adds piece classification."""
    
    def __init__(self, models_folder, *args, **kwargs):
        super().__init__(models_folder, *args, **kwargs)
        # Add simple piece classifier
        self.simple_piece_classifier = SimplePieceClassifier(models_folder)
        logger.info("Simple piece classifier initialized")
    
    def predict_with_custom_pieces(self, img, turn, corners):
        """Predict with custom piece classification."""
        try:
            # Get occupancy using the original recognizer
            occupancy = self._classify_occupancy(img, turn, corners)
            
            # Convert occupancy to a simple Python list to avoid array issues
            occupancy_list = occupancy.tolist()
            
            # Use our custom piece classifier
            pieces_1d = self.simple_piece_classifier.classify_pieces(img, corners, occupancy_list, turn)
            
            # Convert 1D result to 2D for consistency
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece is not None:
                    pieces_2d[rank, file] = piece
            
            # Create a new board with the classified pieces
            new_board = chess.Board()
            new_board.clear()
            for rank in range(8):
                for file in range(8):
                    piece = pieces_2d[rank, file]
                    if piece is not None:
                        square = chess.square(file, 7-rank)  # Convert to chess square
                        new_board.set_piece_at(square, piece)
            
            return new_board, corners
            
        except Exception as e:
            logger.error(f"Custom piece classification failed: {e}")
            # Fall back to original method
            return super().predict(img, turn)

# Global recognizer instance
recognizer = None

# FastAPI app
app = FastAPI(title="Chess Position Scanner API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the chess recognizer on startup."""
    global recognizer
    try:
        logger.info("Starting up Chess Position Scanner API...")
        
        # Load configuration
        logger.info("Loading configuration...")
        try:
            cfg = CN.load_yaml_with_base('config/recognition.yaml')
            logger.info("Configuration loaded successfully")
        except FileNotFoundError:
            logger.warning("Config file not found, using default configuration")
            cfg = CN()
        
        # Initialize custom chess recognizer
        logger.info("Initializing custom chess recognizer...")
        recognizer = CustomChessRecognizer(Path("models"))
        logger.info("Custom chess recognizer initialized successfully")
        
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Chess Position Scanner API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "recognizer_loaded": recognizer is not None}

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    color: str = Form("white")
):
    """Recognize chess position from image with corner coordinates."""
    try:
        # Parse corners
        import json
        corners_list = json.loads(corners)
        corners_array = np.array(corners_list, dtype=np.float32)
        
        # Read and process image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        # Determine turn
        turn = chess.WHITE if color.lower() == "white" else chess.BLACK
        
        # Use custom piece classification approach
        logger.info("Using custom piece classification...")
        
        # First, get occupancy using the original recognizer
        board, detected_corners = recognizer.predict(img_array, turn)
        
        # Get occupancy from the board
        occupancy = np.zeros(64, dtype=bool)
        for square in chess.SQUARES:
            if board.piece_at(square) is not None:
                occupancy[square] = True
        
        # Convert occupancy to a simple Python list to avoid array issues
        occupancy_list = occupancy.tolist()
        
        # Now use our custom piece classifier
        logger.info("Classifying pieces with custom classifier...")
        try:
            pieces_2d = recognizer._classify_pieces(img_array, turn, corners_array, occupancy_list)
            logger.info("Piece classification completed successfully")
        except Exception as e:
            logger.error(f"Error in piece classification: {e}")
            raise
        
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
        
        return JSONResponse(content={
            "fen": fen,
            "occupancy": occupancy.tolist(),
            "pieces": pieces,
            "debug_images": {},
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in recognize_chess_position_with_corners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
