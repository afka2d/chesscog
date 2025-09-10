#!/usr/bin/env python3
"""
Main API with manual corner occupancy detection - exactly matching the working commit.
This uses manual corners only, no automatic corner detection.
"""

import logging
import json
from pathlib import Path
import numpy as np
import chess
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from chesscog.recognition.recognition import ChessRecognizer
from simple_piece_classifier_test import SimplePieceClassifierTest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Position Scanner API - Manual Occupancy + Kings",
    description="API for recognizing chess positions with manual corner occupancy detection and king piece classification for testing.",
    version="1.0.0"
)

# Global instances
piece_classifier: SimplePieceClassifierTest = None
recognizer: ChessRecognizer = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Chess Position Scanner API...")
    logger.info("🧪 TEST MODE: All pieces will be classified as KINGS for verification")
    
    logger.info("Initializing TEST piece classifier...")
    global piece_classifier
    models_dir = Path("models/piece_classifier")
    piece_classifier = SimplePieceClassifierTest(models_dir)
    logger.info("TEST piece classifier initialized successfully")
    
    logger.info("Initializing ChessCog recognizer...")
    global recognizer
    recognizer = ChessRecognizer()
    logger.info("ChessCog recognizer initialized successfully")
    logger.info("Startup completed successfully")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy - Manual Occupancy + Kings",
        "piece_classifier_loaded": piece_classifier is not None,
        "recognizer_loaded": recognizer is not None,
        "test_mode": "All pieces will be classified as KINGS"
    })

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    turn: str = Form("white")
):
    try:
        # Read image bytes and decode with OpenCV (BGR format) like the working commit
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        corners_list = json.loads(corners)
        corners_array = np.array(corners_list, dtype=np.float32)
        
        logger.info("🔍 Using MANUAL corners - no automatic corner detection")
        logger.info(f"Manual corners: {corners_list}")
        
        # Direct call to _classify_occupancy with manual corners (bypassing any automatic detection)
        logger.info("Detecting occupancy with manual corners...")
        occupancy = recognizer._classify_occupancy(img_array, chess.WHITE if turn == "white" else chess.BLACK, corners_array)
        
        # Handle both 1D and 2D occupancy arrays
        if occupancy.ndim == 1:
            occupancy = occupancy.reshape(8, 8)
            logger.info("Converted 1D occupancy array to 2D")
        
        # Count occupied squares
        occupied_count = np.sum(occupancy)
        logger.info(f"Detected {occupied_count} occupied squares out of 64")
        
        logger.info("🧪 TEST MODE: Classifying pieces as KINGS...")
        # Convert occupancy to list format for the piece classifier
        occupancy_list = []
        for rank in range(8):
            for file in range(8):
                occupancy_list.append(bool(occupancy[rank, file]))
        
        pieces_1d = piece_classifier.classify_pieces(img_array, corners_array, occupancy_list, chess.WHITE if turn == "white" else chess.BLACK)
        
        # Convert to 2D array
        pieces_2d = np.full((8, 8), None, dtype=object)
        for i, piece in enumerate(pieces_1d):
            rank, file = i // 8, i % 8
            if piece is not None:
                pieces_2d[rank, file] = piece
        
        # Create new board with detected pieces
        new_board = chess.Board()
        new_board.clear()
        for rank in range(8):
            for file in range(8):
                piece = pieces_2d[rank, file]
                if piece is not None:
                    new_board.set_piece_at(chess.square(file, 7 - rank), piece)
        
        fen = new_board.fen()
        pieces = [str(p) if p else None for p in pieces_1d]
        occupancy_response = [bool(o) for o in occupancy_list]
        
        # Count kings for verification
        kings_count_in_response = sum(1 for p in pieces if p and ('k' in p.lower() or 'king' in p.lower()))
        logger.info(f"🧪 TEST MODE: Generated {kings_count_in_response} KINGS in response")
        
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
