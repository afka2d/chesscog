#!/usr/bin/env python3
"""
Marshall Improved API - Uses improved Marshall models for better accuracy.
This runs on port 8003 and uses:
- Marshall occupancy model (improved)
- Original color classification model (working well)
- Combined piece classification model (Marshall + grey background data)

This API maintains the exact same input/output format as the production API.
"""

import logging
import json
import time
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Position Scanner API - Marshall Improved",
    description="Improved API using Marshall-trained models for better accuracy. Runs on port 8003.",
    version="1.0.0-marshall"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    model = models.resnet18(weights=None)  # ResNet18 for combined piece classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

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
    
    # Ensure the order is: top-left, top-right, bottom-right, bottom-left
    # We need to find which corner is top-left (smallest x+y)
    sums = np.sum(sorted_corners, axis=1)
    top_left_idx = np.argmin(sums)
    
    # Reorder starting from top-left
    reordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
    
    return reordered_corners

def warp_chessboard(img_array, corners_array):
    """Warp chessboard using the exact logic from the working commit."""
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

def extract_square(warped_board, rank, file):
    """Extract a single square from the warped board using exact logic from working commit."""
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

def load_marshall_occupancy_model():
    """Load the Marshall occupancy model (architecture + state_dict)."""
    try:
        # Load original model architecture
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            logger.error(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original occupancy model architecture loaded")
        
        # Load Marshall weights
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            logger.error(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall occupancy weights loaded")
        
        # Apply Marshall weights to original architecture
        model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall occupancy model loaded successfully")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading Marshall occupancy model: {e}")
        return None

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        # Load the combined piece classifier
        model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
        if not model_path.exists():
            logger.error(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        # Create the model architecture first
        model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        logger.info("✅ Combined piece classifier architecture created")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("✅ Combined piece classifier weights loaded")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading combined piece classifier: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Marshall Improved API...")
    logger.info("📍 Running on port 8003 (separate from production)")
    
    global occupancy_model, color_model, piece_type_model
    
    # Load Marshall occupancy model
    logger.info("Loading Marshall occupancy classifier...")
    occupancy_model = load_marshall_occupancy_model()
    if occupancy_model is None:
        logger.error("❌ Failed to load Marshall occupancy model")
        raise RuntimeError("Marshall occupancy model not found")
    logger.info("✅ Marshall occupancy classifier loaded successfully")
    
    # Load original color classifier (working well, no need to change)
    logger.info("Loading original color classifier...")
    color_model_path = Path("models/color_classifier_simple.pt")
    if color_model_path.exists():
        color_model = _get_color_model_architecture(len(COLOR_LABELS))
        color_model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        color_model.eval()
        logger.info("✅ Original color classifier loaded successfully")
    else:
        logger.error(f"Color classifier model not found at {color_model_path}")
        raise RuntimeError("Color classifier model not found")
    
    # Load combined piece type classifier
    logger.info("Loading combined piece type classifier...")
    piece_type_model = load_combined_piece_classifier()
    if piece_type_model is None:
        logger.error("❌ Failed to load combined piece classifier")
        raise RuntimeError("Combined piece classifier not found")
    logger.info("✅ Combined piece type classifier loaded successfully")
    
    logger.info("🎉 Marshall Improved API startup completed successfully")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy - Marshall Improved API",
        "port": 8003,
        "occupancy_model_loaded": occupancy_model is not None,
        "color_model_loaded": color_model is not None,
        "piece_type_model_loaded": piece_type_model is not None,
        "classifier_type": "Marshall Improved Models",
        "environment": "marshall_improved",
        "models": {
            "occupancy": "Marshall-trained ResNet",
            "color": "Original MobileNetV2",
            "piece_type": "Combined ResNet18 (Marshall + Grey Background)"
        }
    })

@app.get("/debug/info")
async def debug_info():
    """Debug endpoint with additional information for development"""
    return JSONResponse(content={
        "api_type": "marshall_improved",
        "port": 8003,
        "models_loaded": {
            "occupancy": occupancy_model is not None,
            "color": color_model is not None,
            "piece_type": piece_type_model is not None
        },
        "model_paths": {
            "occupancy": "models_marshall_improved/occupancy_marshall.pt",
            "color": "models/color_classifier_simple.pt",
            "piece_type": "models_marshall_improved/combined_piece_classifier.pt"
        },
        "labels": {
            "color": COLOR_LABELS,
            "piece_type": PIECE_TYPE_LABELS
        }
    })

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),
    turn: str = Form("white"),
    debug: bool = Form(False)
):
    """
    Recognize chess position from image with given corners using Marshall improved models.
    
    This endpoint uses:
    - Marshall-trained occupancy detection model (improved accuracy)
    - Original color classification model (working well)
    - Combined piece classification model (Marshall + grey background data)
    
    Parameters and response format are identical to the production API.
    """
    try:
        # Decode image
        img_bytes = await image.read()
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Parse corners
        corners_array = json.loads(corners)
        corners_array = np.array(corners_array, dtype=np.float32)
        
        logger.info(f"🔧 Marshall Improved API - Processing image")
        logger.info(f"Manual corners: {corners_array.tolist()}")
        logger.info(f"Debug mode: {debug}")
        
        # Warp chessboard using exact working logic
        warped_board = warp_chessboard(img_array, corners_array)
        logger.info(f"Warped board shape: {warped_board.shape}")
        
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
        
        # Debug information
        debug_info = {
            "squares_processed": 0,
            "occupied_squares": 0,
            "pieces_detected": 0,
            "confidence_scores": [],
            "processing_time": 0,
            "model_info": {
                "occupancy": "Marshall-trained ResNet",
                "color": "Original MobileNetV2", 
                "piece_type": "Combined ResNet18 (Marshall + Grey Background)"
            }
        }
        
        start_time = time.time()
        
        # Process each square using exact working logic
        for rank in range(8):
            for file in range(8):
                square_img = extract_square(warped_board, rank, file)
                debug_info["squares_processed"] += 1
                
                # Occupancy detection using Marshall model
                input_tensor = occupancy_transform(Image.fromarray(square_img)).unsqueeze(0)
                with torch.no_grad():
                    occupancy_output = occupancy_model(input_tensor)
                    probs = torch.softmax(occupancy_output, dim=1)
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
                
                is_occupied = prediction == 1 and confidence > 0.3  # 1 = occupied, 0 = empty
                occupancy_list[rank * 8 + file] = is_occupied
                
                if is_occupied:
                    debug_info["occupied_squares"] += 1
                    
                    # Color classification for occupied squares (using original model)
                    predicted_color = "unknown"
                    predicted_piece_type = "unknown"
                    color_confidence = 0.0
                    piece_confidence = 0.0
                    
                    if color_model and piece_type_model:
                        try:
                            # Color classification
                            input_tensor_color = color_transform(Image.fromarray(square_img)).unsqueeze(0)
                            with torch.no_grad():
                                color_output = color_model(input_tensor_color)
                                color_probs = torch.softmax(color_output, dim=1)
                                color_confidence = torch.max(color_probs).item()
                                predicted_color_idx = torch.argmax(color_output, dim=1).item()
                                predicted_color = COLOR_LABELS[predicted_color_idx]
                            
                            # Piece type classification using combined model
                            input_tensor_piece_type = piece_type_transform(Image.fromarray(square_img)).unsqueeze(0)
                            with torch.no_grad():
                                piece_type_output = piece_type_model(input_tensor_piece_type)
                                piece_type_probs = torch.softmax(piece_type_output, dim=1)
                                piece_type_confidence = torch.max(piece_type_probs).item()
                                predicted_piece_type_idx = torch.argmax(piece_type_output, dim=1).item()
                                predicted_piece_type = PIECE_TYPE_LABELS[predicted_piece_type_idx]
                            
                            # Store confidence scores for debugging
                            debug_info["confidence_scores"].append({
                                "square": f"{chr(97+file)}{8-rank}",
                                "occupancy_confidence": confidence,
                                "color_confidence": color_confidence,
                                "piece_confidence": piece_type_confidence
                            })
                            
                            # Only use high confidence predictions
                            if color_confidence >= 0.7 and piece_type_confidence >= 0.7:
                                square_name = f"{chr(97+file)}{8-rank}"
                                logger.info(f"Square {square_name} occupied - classified as {predicted_color} {predicted_piece_type} (color conf: {color_confidence:.3f}, piece conf: {piece_type_confidence:.3f})")
                                
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
                                debug_info["pieces_detected"] += 1
                            else:
                                square_name = f"{chr(97+file)}{8-rank}"
                                logger.warning(f"Square {square_name} - low confidence: color={color_confidence:.3f}, piece={piece_type_confidence:.3f}")
                        except Exception as e:
                            logger.warning(f"Classification failed for square {chr(97+file)}{8-rank}: {e}")
                    else:
                        logger.warning(f"Models not loaded. Cannot classify piece for square {chr(97+file)}{8-rank}")
        
        debug_info["processing_time"] = time.time() - start_time
        
        # Generate response
        fen = board.fen()
        pieces_response = [str(p) if p else None for p in pieces_1d]
        occupancy_response = [bool(o) for o in occupancy_list]
        
        total_pieces = sum(1 for p in pieces_1d if p is not None)
        logger.info(f"Total pieces found: {total_pieces}")
        
        response = {
            "fen": fen,
            "pieces": pieces_response,
            "occupancy": occupancy_response,
            "success": True
        }
        
        # Add debug information if requested
        if debug:
            response["debug_info"] = debug_info
            response["api_info"] = {
                "type": "marshall_improved",
                "port": 8003,
                "version": "1.0.0-marshall"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in recognize_chess_position_with_corners: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Marshall Improved API on port 8003")
    print("📍 Production API runs on port 8000")
    print("🔧 Local Development API runs on port 8001")
    print("🎯 This API uses improved Marshall models for better accuracy")
    uvicorn.run(app, host="0.0.0.0", port=8003)