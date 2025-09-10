#!/usr/bin/env python3
"""
Simple main.py that adds piece classification to your existing working API.
This preserves your working corner detection and occupancy classification.
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
    
    def _classify_pieces(self, img, turn, corners, occupancy):
        """Classify pieces using the simple piece classifier."""
        logger.debug(f"Classifying pieces with occupancy shape: {occupancy.shape}")
        
        try:
            logger.info("Using simple piece classification")
            
            # Convert occupancy to list if it's a numpy array
            if hasattr(occupancy, 'tolist'):
                occupancy_list = occupancy.tolist()
            else:
                occupancy_list = list(occupancy)
            
            # Use the simple piece classifier
            pieces_1d = self.simple_piece_classifier.classify_pieces(img, corners, occupancy_list, turn)
            
            # Convert 1D result to 2D for consistency
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece is not None:
                    pieces_2d[rank, file] = piece
            
            logger.info("Simple piece classification completed successfully")
            return pieces_2d
            
        except Exception as e:
            logger.error(f"Simple piece classification failed: {e}")
            logger.warning("Falling back to parent method")
            # Fall back to parent method
            pieces_1d = super()._classify_pieces(img, turn, corners, occupancy)
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece is not None:
                    pieces_2d[rank, file] = piece
            return pieces_2d

def encode_image(image, max_width=800, max_height=600):
    """Encode image to base64 string with size constraints."""
    try:
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Resize if too large
        width, height = image.size
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

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
        
        # Recognize chess position
        logger.info("Recognizing chess position...")
        board, detected_corners = recognizer.predict(img_array, turn)
        
        # Convert board to FEN
        fen = board.fen()
        
        # Get occupancy from the board
        occupancy = np.zeros(64, dtype=bool)
        for square in chess.SQUARES:
            if board.piece_at(square) is not None:
                occupancy[square] = True
        
        # Get pieces as strings
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                pieces.append(piece_name)
            else:
                pieces.append(None)
        
        # Convert pieces to 2D array for consistency
        pieces_2d = np.full((8, 8), None, dtype=object)
        for i, piece in enumerate(pieces):
            rank, file = i // 8, i % 8
            if piece:
                # Convert piece name to chess.Piece object
                if piece.startswith('white_'):
                    color = chess.WHITE
                    piece_type = piece[6:]  # Remove 'white_' prefix
                else:
                    color = chess.BLACK
                    piece_type = piece[6:]  # Remove 'black_' prefix
                
                piece_map = {
                    'pawn': chess.PAWN, 'rook': chess.ROOK, 'knight': chess.KNIGHT,
                    'bishop': chess.BISHOP, 'queen': chess.QUEEN, 'king': chess.KING
                }
                
                if piece_type in piece_map:
                    pieces_2d[rank, file] = chess.Piece(piece_map[piece_type], color)
        
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
