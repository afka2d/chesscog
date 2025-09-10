#!/usr/bin/env python3
"""
Main API with working occupancy detection and king piece classifier for testing.
"""

import logging
import json
from pathlib import Path
import numpy as np
import chess
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
    title="Chess Position Scanner API - Working Occupancy + Kings",
    description="API for recognizing chess positions with working occupancy detection and king piece classification for testing.",
    version="1.0.0"
)

# Global instances
piece_classifier: SimplePieceClassifierTest = None
occupancy_recognizer: ChessRecognizer = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Chess Position Scanner API...")
    logger.info("🧪 TEST MODE: All pieces will be classified as KINGS for verification")
    
    logger.info("Initializing TEST piece classifier...")
    global piece_classifier
    models_dir = Path("models/piece_classifier")
    piece_classifier = SimplePieceClassifierTest(models_dir)
    logger.info("TEST piece classifier initialized successfully")
    
    logger.info("Initializing occupancy recognizer...")
    global occupancy_recognizer
    occupancy_recognizer = ChessRecognizer()
    logger.info("Occupancy recognizer initialized successfully")
    logger.info("Startup completed successfully")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy - Working Occupancy + Kings",
        "piece_classifier_loaded": piece_classifier is not None,
        "occupancy_recognizer_loaded": occupancy_recognizer is not None,
        "test_mode": "All pieces will be classified as KINGS"
    })

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    turn: str = Form("white")
):
    try:
        img_array = np.array(Image.open(image.file).convert("RGB"))
        corners_list = json.loads(corners)
        
        logger.info("Detecting occupancy with working ChessCog method...")
        # Use ChessCog's recognizer for occupancy (the working method)
        board, occupancy_result = occupancy_recognizer.predict(img_array, turn)
        occupancy = [bool(o) for o in occupancy_result]
        logger.info(f"Detected {sum(occupancy)} occupied squares out of 64")
        
        logger.info("🧪 TEST MODE: Classifying pieces as KINGS...")
        pieces_1d = piece_classifier.classify_pieces(img_array, np.array(corners_list), occupancy, chess.WHITE if turn == "white" else chess.BLACK)
        
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
        occupancy_response = [bool(o) for o in occupancy]
        
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
