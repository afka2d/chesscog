#!/usr/bin/env python3
"""
Main API with two-stage piece classifier:
1. Color classifier (white vs black)
2. Piece type classifier (6 types per color)

This should achieve 70%+ accuracy by simplifying the problem.
"""

import logging
import json
from pathlib import Path
import numpy as np
import chess
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Position Scanner API - Two-Stage Classifier",
    description="API for recognizing chess positions with two-stage piece classification (color + piece type).",
    version="1.0.0"
)

# Global instances
occupancy_model = None
color_model = None
piece_model = None

class TwoStagePieceClassifier:
    def __init__(self, color_model_path, piece_model_path):
        self.color_model = torch.load(color_model_path, map_location='cpu', weights_only=False)
        self.piece_model = torch.load(piece_model_path, map_location='cpu', weights_only=False)
        self.color_model.eval()
        self.piece_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_piece(self, square_img):
        """Predict piece using two-stage approach"""
        # Convert BGR to RGB
        square_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(square_rgb)
        
        # Apply transforms
        img_tensor = self.transform(pil_img).unsqueeze(0)
        
        # Stage 1: Color classification
        with torch.no_grad():
            color_output = self.color_model(img_tensor)
            color_pred = color_output.argmax(dim=1).item()
            color_conf = torch.softmax(color_output, dim=1).max().item()
        
        # Stage 2: Piece type classification
        with torch.no_grad():
            piece_output = self.piece_model(img_tensor)
            piece_pred = piece_output.argmax(dim=1).item()
            piece_conf = torch.softmax(piece_output, dim=1).max().item()
        
        # Combine predictions
        color = "white" if color_pred == 0 else "black"
        piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        piece_type = piece_types[piece_pred]
        
        # Convert to chess piece
        piece = self.piece_name_to_chess_piece(f"{color}_{piece_type}")
        combined_conf = color_conf * piece_conf
        
        return piece, combined_conf, color_conf, piece_conf
    
    def piece_name_to_chess_piece(self, piece_name):
        """Convert piece name to chess.Piece object"""
        color_str, piece_type_str = piece_name.split('_')
        color = chess.WHITE if color_str == 'white' else chess.BLACK
        
        piece_type_map = {
            'pawn': chess.PAWN,
            'rook': chess.ROOK,
            'knight': chess.KNIGHT,
            'bishop': chess.BISHOP,
            'queen': chess.QUEEN,
            'king': chess.KING
        }
        
        return chess.Piece(piece_type_map[piece_type_str], color)

def warp_chessboard(img, corners):
    """Warp chessboard image to 800x800 with 100x100 squares"""
    target_size = 800
    target_corners = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    warped = cv2.warpPerspective(img, M, (target_size, target_size))
    return warped

def extract_square(warped_img, rank, file, square_size=100):
    """Extract a square from the warped chessboard"""
    y_start = rank * square_size
    y_end = y_start + square_size
    x_start = file * square_size
    x_end = x_start + square_size
    
    square = warped_img[y_start:y_end, x_start:x_end]
    return square

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Chess Position Scanner API with Two-Stage Classifier...")
    
    # Load occupancy classifier
    logger.info("Loading occupancy classifier...")
    global occupancy_model
    occupancy_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    occupancy_model = torch.load(str(occupancy_model_path), map_location='cpu', weights_only=False)
    occupancy_model.eval()
    logger.info("Occupancy classifier loaded successfully")
    
    # Load two-stage piece classifier
    logger.info("Loading two-stage piece classifier...")
    global color_model, piece_model
    color_model_path = Path("models/color_classifier_best.pt")
    piece_model_path = Path("models/piece_type_classifier_best.pt")
    
    if not color_model_path.exists() or not piece_model_path.exists():
        logger.error("Two-stage classifier models not found. Please run create_two_stage_classifier.py first.")
        raise RuntimeError("Two-stage classifier models not found")
    
    piece_classifier = TwoStagePieceClassifier(color_model_path, piece_model_path)
    color_model = piece_classifier.color_model
    piece_model = piece_classifier.piece_model
    logger.info("Two-stage piece classifier loaded successfully")
    
    logger.info("Startup completed successfully")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy - Two-Stage Classifier",
        "occupancy_model_loaded": occupancy_model is not None,
        "color_model_loaded": color_model is not None,
        "piece_model_loaded": piece_model is not None,
        "classifier_type": "Two-Stage (Color + Piece Type)",
        "expected_accuracy": "70%+"
    })

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    turn: str = Form("white")
):
    try:
        # Decode image
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Parse corners
        corners_list = json.loads(corners)
        corners_array = np.array(corners_list, dtype=np.float32)
        
        logger.info("🔧 Using two-stage piece classification")
        logger.info(f"Manual corners: {corners_list}")
        
        # Warp chessboard
        warped_board = warp_chessboard(img_array, corners_array)
        logger.info(f"Warped board shape: {warped_board.shape}")
        
        # Detect occupancy
        logger.info("Detecting occupancy...")
        occupancy = occupancy_model._classify_occupancy(img_array, chess.WHITE if turn == "white" else chess.BLACK, corners_array)
        
        if occupancy.ndim == 1:
            occupancy = occupancy.reshape(8, 8)
            logger.info("Converted 1D occupancy array to 2D")
        
        occupied_count = np.sum(occupancy)
        logger.info(f"Detected {occupied_count} occupied squares out of 64")
        
        # Classify pieces using two-stage approach
        logger.info("Classifying pieces using two-stage approach...")
        board = chess.Board()
        board.clear()
        
        piece_classifier = TwoStagePieceClassifier("models/color_classifier_best.pt", "models/piece_type_classifier_best.pt")
        
        pieces_found = 0
        for rank in range(8):
            for file in range(8):
                is_occupied = occupancy[rank, file]
                if is_occupied:
                    # Extract square
                    square = extract_square(warped_board, rank, file)
                    
                    # Predict piece using two-stage approach
                    piece, combined_conf, color_conf, piece_conf = piece_classifier.predict_piece(square)
                    
                    if combined_conf > 0.5:  # Confidence threshold
                        logger.info(f"Square {chr(97+file)}{8-rank} occupied (conf: {combined_conf:.3f}) - classified as {piece} (color: {color_conf:.3f}, type: {piece_conf:.3f})")
                        
                        # Convert to chess square
                        square = chess.square(file, 7 - rank)
                        board.set_piece_at(square, piece)
                        pieces_found += 1
        
        logger.info(f"Total pieces found: {pieces_found}")
        
        # Generate FEN and response
        fen = board.fen()
        pieces = []
        occupancy_list = []
        
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                pieces.append(str(piece) if piece else None)
                occupancy_list.append(bool(occupancy[rank, file]))
        
        return {
            "fen": fen,
            "pieces": pieces,
            "occupancy": occupancy_list,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in recognize_chess_position_with_corners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
