#!/usr/bin/env python3
"""
Main API with exact occupancy detection logic from commit cb0a8f631c3b975d7a61e51dc040a576835ad324
This replicates the working occupancy detection exactly as it was in that commit.
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
    title="Chess Position Scanner API - Exact Working Occupancy",
    description="API for recognizing chess positions with exact working occupancy detection from commit cb0a8f631c3b975d7a61e51dc040a576835ad324.",
    version="1.0.0"
)

# Global instances
occupancy_model = None
piece_model = None

# Piece class mapping
PIECE_CLASSES = [
    'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
    'black_queen', 'black_rook', 'white_bishop', 'white_king', 
    'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]

def sort_corner_points(corners):
    """Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left."""
    # Convert to numpy array if needed
    corners = np.array(corners, dtype=np.float32)
    
    # Find center
    center = np.mean(corners, axis=0)
    
    # Sort by angle from center
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    # Reorder corners
    sorted_corners = corners[sorted_indices]
    
    return sorted_corners

def warp_chessboard(img, corners):
    """Warp the chessboard using manual corners - exact logic from working commit."""
    # Sort corners
    sorted_corners = sort_corner_points(corners)
    
    # Define target size (8x8 squares, each 100x100 pixels)
    target_size = (800, 800)
    
    # Define target corners (top-left, top-right, bottom-right, bottom-left)
    target_corners = np.array([
        [0, 0],           # top-left
        [target_size[0], 0],  # top-right
        [target_size[0], target_size[1]],  # bottom-right
        [0, target_size[1]]   # bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform
    transform_matrix = cv2.getPerspectiveTransform(sorted_corners, target_corners)
    
    # Warp the image
    warped = cv2.warpPerspective(img, transform_matrix, target_size)
    
    return warped

def extract_square(warped_board, rank, file):
    """Extract a specific square from the warped board - exact logic from working commit."""
    # Calculate square coordinates (100x100 pixels each)
    x1 = file * 100
    y1 = rank * 100
    x2 = x1 + 100
    y2 = y1 + 100
    
    # Extract square
    square = warped_board[y1:y2, x1:x2]
    
    return square

def predict_occupancy(square_img, occupancy_model, occupancy_transform):
    """Predict if a square is occupied - exact logic from working commit."""
    with torch.no_grad():
        # Apply transform
        input_tensor = occupancy_transform(square_img).unsqueeze(0)
        
        # Get prediction
        output = occupancy_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        return prediction == 1, confidence  # 1 = occupied, 0 = empty

def predict_piece(square_img, piece_model, piece_transform):
    """Predict the piece type on an occupied square - exact logic from working commit."""
    with torch.no_grad():
        # Apply transform
        input_tensor = piece_transform(square_img).unsqueeze(0)
        
        # Get prediction
        output = piece_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        return PIECE_CLASSES[prediction], confidence

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Chess Position Scanner API...")
    logger.info("🔧 Using EXACT working occupancy logic from commit cb0a8f631c3b975d7a61e51dc040a576835ad324")
    logger.info("🎯 Using ResNet_lightweight piece classifier for real piece classification")

    # Load occupancy classifier - exact path from working commit
    logger.info("Loading occupancy classifier...")
    global occupancy_model
    occupancy_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    occupancy_model = torch.load(str(occupancy_model_path), map_location='cpu', weights_only=False)
    occupancy_model.eval()
    logger.info("Occupancy classifier loaded successfully")

    # Load piece classifier - using ResNet_uniform (best available model)
    logger.info("Loading ResNet_uniform piece classifier...")
    global piece_model
    piece_model_path = Path("runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
    piece_model = torch.load(str(piece_model_path), map_location='cpu', weights_only=False)
    piece_model.eval()
    logger.info("ResNet_uniform piece classifier loaded successfully")

    logger.info("Startup completed successfully")

@app.get("/health")
async def health_check():
        return JSONResponse(content={
            "status": "healthy - Exact Working Occupancy + Real Piece Classification",
            "occupancy_model_loaded": occupancy_model is not None,
            "piece_model_loaded": piece_model is not None,
            "piece_classifier": "ResNet_uniform (best available - 14.8% accuracy)",
            "logic_source": "Commit cb0a8f631c3b975d7a61e51dc040a576835ad324"
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
        
        logger.info("🔧 Using EXACT working occupancy logic from commit cb0a8f631c3b975d7a61e51dc040a576835ad324")
        logger.info(f"Manual corners: {corners_list}")
        
        # Define transforms - exact from working commit
        occupancy_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),  # Match training config
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        piece_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 200)),  # Match training config
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Warp the chessboard - exact logic from working commit
        warped_board = warp_chessboard(img_array, corners_array)
        logger.info(f"Warped board shape: {warped_board.shape}")
        
        # Create chess board
        board = chess.Board()
        board.clear()  # Start with empty board
        
        pieces_found = 0
        occupancy_list = []
        
        # Process each square - exact logic from working commit
        for rank in range(8):
            for file in range(8):
                # Extract square
                square_img = extract_square(warped_board, rank, file)
                
                # Check occupancy
                is_occupied, occupancy_conf = predict_occupancy(square_img, occupancy_model, occupancy_transform)
                
                # Store occupancy for API response
                occupancy_list.append(is_occupied and occupancy_conf > 0.5)
                
                if is_occupied and occupancy_conf > 0.5:  # Confidence threshold
                    # Predict piece type using ResNet_lightweight model
                    piece_type, piece_conf = predict_piece(square_img, piece_model, piece_transform)
                    
                    if piece_conf > 0.3:  # Confidence threshold for piece classification
                        logger.info(f"Square {chr(97+file)}{8-rank} occupied (conf: {occupancy_conf:.3f}) - classified as {piece_type} (conf: {piece_conf:.3f})")
                        
                        # Convert to chess square
                        square = chess.square(file, 7 - rank)  # Convert to chess coordinates
                        
                        # Create piece from classification
                        if piece_type.startswith('white_'):
                            color = chess.WHITE
                            piece_name = piece_type[6:]  # Remove 'white_' prefix
                        else:
                            color = chess.BLACK
                            piece_name = piece_type[6:]  # Remove 'black_' prefix
                        
                        # Map piece names to chess constants
                        piece_map = {
                            'pawn': chess.PAWN,
                            'knight': chess.KNIGHT,
                            'bishop': chess.BISHOP,
                            'rook': chess.ROOK,
                            'queen': chess.QUEEN,
                            'king': chess.KING
                        }
                        
                        piece = chess.Piece(piece_map[piece_name], color)
                        board.set_piece_at(square, piece)
                        pieces_found += 1
                    else:
                        logger.info(f"Square {chr(97+file)}{8-rank} occupied but piece classification confidence too low ({piece_conf:.3f})")
        
        logger.info(f"Total pieces found: {pieces_found}")
        
        # Generate FEN
        fen = board.fen()
        
        # Generate pieces list for API response
        pieces = []
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                if piece:
                    pieces.append(piece.symbol())
                else:
                    pieces.append(None)
        
        pieces_count = sum(1 for p in pieces if p is not None)
        logger.info(f"Generated {pieces_count} pieces in response")

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
