#!/usr/bin/env python3
"""
Production API for Chess Position Scanner
Deployed at: https://api.chesspositionscanner.store/recognize_chess_position_with_corners
"""

import logging
import json
import os
from pathlib import Path
import numpy as np
import chess
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from torchvision import transforms, models
import torch.nn as nn
import traceback
from datetime import datetime

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chess_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app with production settings
app = FastAPI(
    title="Chess Position Scanner API",
    description="Production API for chess position recognition with occupancy detection, color classification, and piece type classification.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
occupancy_model = None
color_model = None
piece_type_model = None

# Color labels (must match training)
COLOR_LABELS = {0: "white", 1: "black"}

# Piece type labels (must match training)
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}

# Helper to get color model architecture (must match training script)
def _get_color_model_architecture(num_classes):
    model = models.mobilenet_v2(weights=None)  # No pre-trained weights for loading state_dict
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# Helper to get piece type model architecture (must match training script)
def _get_piece_type_model_architecture(num_classes):
    model = models.efficientnet_b0(weights=None)  # No pre-trained weights for loading state_dict
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def sort_corner_points(corners):
    """Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left."""
    try:
        # Convert to numpy array if needed
        corners = np.array(corners, dtype=np.float32)
        
        # Find center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder corners
        sorted_corners = corners[sorted_indices]
        
        # Ensure the order is: top-left, top-right, bottom-right, bottom-left
        # We need to find which corner is top-left (smallest x+y)
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        reordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return reordered_corners
    except Exception as e:
        logger.error(f"Error in sort_corner_points: {e}")
        raise

def warp_chessboard(img_array, corners_array):
    """Warp chessboard using the exact logic from the working commit."""
    try:
        # Sort corners to ensure correct order
        corners = sort_corner_points(corners_array)
        
        # Define destination points for a square board
        board_size = 800
        dst_points = np.array([
            [0, 0],                           # top-left
            [board_size - 1, 0],             # top-right
            [board_size - 1, board_size - 1], # bottom-right
            [0, board_size - 1]              # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        M = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img_array, M, (board_size, board_size))
        
        return warped
    except Exception as e:
        logger.error(f"Error in warp_chessboard: {e}")
        raise

def extract_square(warped_board, rank, file):
    """Extract a single square from the warped board using exact logic from working commit."""
    try:
        board_size = warped_board.shape[0]
        square_size = board_size // 8
        
        # Calculate square boundaries
        x1 = file * square_size
        y1 = rank * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Extract square
        square = warped_board[y1:y2, x1:x2]
        
        return square
    except Exception as e:
        logger.error(f"Error in extract_square: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the API and load all models."""
    try:
        logger.info("🚀 Starting Chess Position Scanner API - Production Version")
        logger.info(f"Startup time: {datetime.now().isoformat()}")
        
        global occupancy_model, color_model, piece_type_model
        
        # Load the exact working occupancy model
        logger.info("Loading working occupancy classifier...")
        occupancy_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not occupancy_model_path.exists():
            raise FileNotFoundError(f"Occupancy model not found at {occupancy_model_path}")
        
        occupancy_model = torch.load(str(occupancy_model_path), map_location='cpu', weights_only=False)
        occupancy_model.eval()
        logger.info("✅ Working occupancy classifier loaded successfully")
        
        # Load color classifier
        logger.info("Loading color classifier...")
        color_model_path = Path("models/color_classifier_simple.pt")
        if not color_model_path.exists():
            raise FileNotFoundError(f"Color classifier model not found at {color_model_path}")
        
        color_model = _get_color_model_architecture(len(COLOR_LABELS))
        color_model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        color_model.eval()
        logger.info("✅ Color classifier loaded successfully")
        
        # Load piece type classifier
        logger.info("Loading piece type classifier...")
        piece_type_model_path = Path("models/piece_classifier_simple.pt")
        if not piece_type_model_path.exists():
            raise FileNotFoundError(f"Piece type classifier model not found at {piece_type_model_path}")
        
        piece_type_model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        piece_type_model.load_state_dict(torch.load(str(piece_type_model_path), map_location='cpu'))
        piece_type_model.eval()
        logger.info("✅ Piece type classifier loaded successfully")
        
        logger.info("🎉 Production API startup completed successfully")
        logger.info(f"API ready at: https://api.chesspositionscanner.store/recognize_chess_position_with_corners")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        logger.error(traceback.format_exc())
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Chess Position Scanner API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "recognize": "/recognize_chess_position_with_corners",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "occupancy_model_loaded": occupancy_model is not None,
                "color_model_loaded": color_model is not None,
                "piece_type_model_loaded": piece_type_model is not None
            },
            "version": "1.0.0",
            "uptime": "operational"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    turn: str = Form("white")
):
    """
    Recognize chess position from image with manual corner coordinates.
    
    Args:
        image: Chess board image file
        corners: JSON string of corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        turn: Current player turn ("white" or "black")
    
    Returns:
        JSON with FEN notation, piece positions, and occupancy data
    """
    start_time = datetime.now()
    request_id = f"req_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
    
    try:
        logger.info(f"🔍 [{request_id}] Processing chess position recognition request")
        logger.info(f"📊 [{request_id}] Image: {image.filename}, Size: {image.size if hasattr(image, 'size') else 'unknown'}")
        logger.info(f"📐 [{request_id}] Turn: {turn}")
        
        # Validate inputs
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        if not corners:
            raise HTTPException(status_code=400, detail="No corner coordinates provided")
        
        # Decode image
        try:
            img_bytes = await image.read()
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_array is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            logger.info(f"🖼️ [{request_id}] Image decoded successfully: {img_array.shape}")
        except Exception as e:
            logger.error(f"❌ [{request_id}] Image decoding failed: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        
        # Parse corners
        try:
            corners_array = json.loads(corners)
            corners_array = np.array(corners_array, dtype=np.float32)
            if corners_array.shape != (4, 2):
                raise ValueError("Corners must be 4 points with 2 coordinates each")
            logger.info(f"📐 [{request_id}] Corners parsed successfully: {corners_array.tolist()}")
        except Exception as e:
            logger.error(f"❌ [{request_id}] Corner parsing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid corner format: {str(e)}")
        
        # Warp chessboard using exact working logic
        try:
            warped_board = warp_chessboard(img_array, corners_array)
            logger.info(f"🔄 [{request_id}] Board warped successfully: {warped_board.shape}")
        except Exception as e:
            logger.error(f"❌ [{request_id}] Board warping failed: {e}")
            raise HTTPException(status_code=400, detail=f"Board warping failed: {str(e)}")
        
        # Initialize board
        board = chess.Board()
        board.clear()
        pieces_1d = [None] * 64
        occupancy_list = [False] * 64
        
        # Transforms for occupancy detection (exact from working commit)
        occupancy_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transforms for color classification
        color_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transforms for piece type classification
        piece_type_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Process each square using exact working logic
        occupied_squares = 0
        classified_pieces = 0
        
        for rank in range(8):
            for file in range(8):
                try:
                    square_img = extract_square(warped_board, rank, file)
                    
                    # Occupancy detection using exact working method
                    input_tensor = occupancy_transform(Image.fromarray(square_img)).unsqueeze(0)
                    with torch.no_grad():
                        occupancy_output = occupancy_model(input_tensor)
                        probs = torch.softmax(occupancy_output, dim=1)
                        prediction = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][prediction].item()
                    
                    is_occupied = prediction == 1 and confidence > 0.5  # 1 = occupied, 0 = empty
                    occupancy_list[rank * 8 + file] = is_occupied
                    
                    if is_occupied:
                        occupied_squares += 1
                        
                        # Color classification for occupied squares
                        predicted_color = "unknown"
                        predicted_piece_type = "unknown"
                        
                        try:
                            # Color classification
                            input_tensor_color = color_transform(Image.fromarray(square_img)).unsqueeze(0)
                            with torch.no_grad():
                                color_output = color_model(input_tensor_color)
                                color_probs = torch.softmax(color_output, dim=1)
                                color_confidence = torch.max(color_probs).item()
                                predicted_color_idx = torch.argmax(color_output, dim=1).item()
                                predicted_color = COLOR_LABELS[predicted_color_idx]
                            
                            # Piece type classification
                            input_tensor_piece_type = piece_type_transform(Image.fromarray(square_img)).unsqueeze(0)
                            with torch.no_grad():
                                piece_type_output = piece_type_model(input_tensor_piece_type)
                                piece_type_probs = torch.softmax(piece_type_output, dim=1)
                                piece_type_confidence = torch.max(piece_type_probs).item()
                                predicted_piece_type_idx = torch.argmax(piece_type_output, dim=1).item()
                                predicted_piece_type = PIECE_TYPE_LABELS[predicted_piece_type_idx]
                            
                            # Only use high confidence predictions
                            if color_confidence >= 0.7 and piece_type_confidence >= 0.7:
                                square_name = f"{chr(97+file)}{8-rank}"
                                logger.info(f"♟️ [{request_id}] Square {square_name}: {predicted_color} {predicted_piece_type} (conf: {color_confidence:.3f}/{piece_type_confidence:.3f})")
                                
                                # Create piece
                                square = chess.square(file, 7 - rank)
                                color_enum = chess.WHITE if predicted_color == "white" else chess.BLACK
                                
                                # Map piece type to chess constants
                                piece_map = {
                                    'pawn': chess.PAWN,
                                    'knight': chess.KNIGHT,
                                    'bishop': chess.BISHOP,
                                    'rook': chess.ROOK,
                                    'queen': chess.QUEEN,
                                    'king': chess.KING
                                }
                                
                                piece_type_enum = piece_map[predicted_piece_type]
                                piece = chess.Piece(piece_type_enum, color_enum)
                                
                                board.set_piece_at(square, piece)
                                pieces_1d[rank * 8 + file] = piece
                                classified_pieces += 1
                            else:
                                square_name = f"{chr(97+file)}{8-rank}"
                                logger.warning(f"⚠️ [{request_id}] Square {square_name}: Low confidence (color: {color_confidence:.3f}, piece: {piece_type_confidence:.3f})")
                                
                        except Exception as e:
                            square_name = f"{chr(97+file)}{8-rank}"
                            logger.warning(f"⚠️ [{request_id}] Classification failed for square {square_name}: {e}")
                            
                except Exception as e:
                    square_name = f"{chr(97+file)}{8-rank}"
                    logger.warning(f"⚠️ [{request_id}] Processing failed for square {square_name}: {e}")
        
        # Generate response
        try:
            fen = board.fen()
            pieces_response = [str(p) if p else None for p in pieces_1d]
            occupancy_response = [bool(o) for o in occupancy_list]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ [{request_id}] Processing completed successfully")
            logger.info(f"📊 [{request_id}] Results: {occupied_squares} occupied squares, {classified_pieces} pieces classified")
            logger.info(f"⏱️ [{request_id}] Processing time: {processing_time:.2f}s")
            
            return {
                "fen": fen,
                "pieces": pieces_response,
                "occupancy": occupancy_response,
                "success": True,
                "metadata": {
                    "request_id": request_id,
                    "processing_time_seconds": processing_time,
                    "occupied_squares": occupied_squares,
                    "classified_pieces": classified_pieces,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Response generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ [{request_id}] Unexpected error: {e}")
        logger.error(f"❌ [{request_id}] Traceback: {traceback.format_exc()}")
        logger.error(f"⏱️ [{request_id}] Processing time before error: {processing_time:.2f}s")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        log_level="info",
        access_log=True
    )
