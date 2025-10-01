#!/usr/bin/env python3
"""
Working Marshall API - Uses existing working models on port 8003.
This provides the same interface as production but runs on a different port.
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
    title="Chess Position Scanner API - Marshall Port",
    description="Working API using existing models on port 8003 for testing.",
    version="1.0.0-marshall-port"
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
    # Use EfficientNet-B0 (same as original API, not ResNet18)
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
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
    """Warp chessboard using simplified approach that matches training."""
    # Use corners as-is (annotation corners are already in correct order)
    corners = np.array(corners_array, dtype=np.float32)
    
    # Define destination points for a square board (simple approach)
    board_size = 800
    dst_points = np.array([
        [0, 0],                    # top-left
        [board_size, 0],           # top-right
        [board_size, board_size],  # bottom-right
        [0, board_size]            # bottom-left
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
    """Load the Marshall occupancy model with correct architecture."""
    try:
        # Create the correct architecture (ResNet18 with 2 outputs)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 outputs for occupancy
        logger.info("✅ Marshall occupancy model architecture created (ResNet18, 2 outputs)")
        
        # Load the Marshall weights (state_dict)
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            logger.error(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall occupancy weights loaded")
        
        # Remove 'model.' prefix from keys if present
        new_state_dict = {}
        for k, v in marshall_weights.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        logger.info("✅ Marshall occupancy model weights loaded successfully")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading Marshall occupancy model: {e}")
        return None

def load_original_piece_classifier():
    """Load the ORIGINAL piece classification model (EfficientNet-B0, production model)."""
    try:
        model_path = Path("models/piece_classifier_simple.pt")
        if not model_path.exists():
            logger.error(f"❌ Original piece classifier not found at {model_path}")
            return None
        
        # Create EfficientNet-B0 architecture (same as production)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(PIECE_TYPE_LABELS))
        logger.info("✅ Original piece classifier architecture created (EfficientNet-B0)")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("✅ Original piece classifier weights loaded")
        
        model.eval()
        logger.info("✅ Original piece classifier loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading original piece classifier: {e}")
        return None

def load_balanced_piece_classifier():
    """Load the BALANCED piece classification model (EfficientNet-B0, trained on both datasets)."""
    try:
        model_path = Path("models_marshall_improved/piece_classifier_balanced.pt")
        if not model_path.exists():
            logger.error(f"❌ Balanced piece classifier not found at {model_path}")
            return None
        
        # Create EfficientNet-B0 architecture (same as original)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(PIECE_TYPE_LABELS))
        logger.info("✅ Balanced piece classifier architecture created (EfficientNet-B0)")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("✅ Balanced piece classifier weights loaded")
        
        model.eval()
        logger.info("✅ Balanced piece classifier loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading balanced piece classifier: {e}")
        return None

def load_combined_piece_classifier():
    """Load the COMBINED piece classification model (trained on both datasets)."""
    try:
        # Load the combined piece classifier (ResNet18 trained on Marshall + Grey data)
        model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
        if not model_path.exists():
            logger.error(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        # Create ResNet18 architecture (combined model uses ResNet18, not EfficientNet)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(PIECE_TYPE_LABELS))
        logger.info("✅ Combined piece classifier architecture created (ResNet18)")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("✅ Combined piece classifier weights loaded")
        
        model.eval()
        logger.info("✅ Combined piece classifier loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading combined piece classifier: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Marshall Improved API...")
    logger.info("📍 Running on port 8003 (using improved Marshall models)")
    
    global occupancy_model, color_model, piece_type_model
    
    # Load Marshall occupancy model
    logger.info("Loading Marshall occupancy classifier...")
    occupancy_model = load_marshall_occupancy_model()
    if occupancy_model is None:
        logger.error("❌ Failed to load Marshall occupancy model")
        raise RuntimeError("Marshall occupancy model not found")
    logger.info("✅ Marshall occupancy classifier loaded successfully")
    
    # Load ORIGINAL color classifier (Marshall and Original perform identically)
    logger.info("Loading ORIGINAL color classifier (same as production)...")
    color_model_path = Path("models/color_classifier_simple.pt")
    if color_model_path.exists():
        color_model = _get_color_model_architecture(len(COLOR_LABELS))
        color_model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        color_model.eval()
        logger.info("✅ ORIGINAL color classifier loaded successfully")
    else:
        logger.error(f"Color classifier model not found at {color_model_path}")
        raise RuntimeError("Color classifier model not found")
    
    # Load BALANCED piece type classifier (trained on both Marshall + Grey datasets)
    logger.info("Loading BALANCED piece type classifier (trained on both datasets)...")
    piece_type_model = load_balanced_piece_classifier()
    if piece_type_model is None:
        logger.error("❌ Failed to load balanced piece classifier")
        raise RuntimeError("Balanced piece classifier not found")
    logger.info("✅ BALANCED piece type classifier loaded successfully")
    
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
            "piece_type": "Balanced EfficientNet-B0 (Marshall + Grey Background)"
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
        # Decode image (keep as BGR for warping - matches training)
        img_bytes = await image.read()
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        # DO NOT convert to RGB before warping - this changes interpolation
        
        # Parse corners
        corners_array = json.loads(corners)
        corners_array = np.array(corners_array, dtype=np.float32)
        
        logger.info(f"🔧 Marshall Improved API - Processing image")
        logger.info(f"Manual corners: {corners_array.tolist()}")
        logger.info(f"Debug mode: {debug}")
        
        # Warp chessboard using simplified logic (no corner sorting, correct dest points)
        warped_board = warp_chessboard(img_array, corners_array)
        logger.info(f"Warped board shape: {warped_board.shape}")
        
        # Initialize board
        board = chess.Board()
        board.clear()
        pieces_1d = [None] * 64
        occupancy_list = [False] * 64
        
        # Transforms for occupancy detection (matching Marshall training - simple 0-1 normalization)
        def preprocess_square_for_occupancy(square_img):
            """Preprocess square for occupancy detection (Marshall training preprocessing)"""
            import cv2
            import numpy as np
            square = cv2.resize(square_img, (100, 100))
            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square = square.astype(np.float32) / 255.0
            square = torch.from_numpy(square).permute(2, 0, 1)
            return square
        
        # Transforms for color classification (matching original training - ImageNet normalization)
        color_transform = transforms.Compose([
            transforms.Resize(64),  # Original uses 64x64
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Transforms for piece type classification (matching balanced training - ImageNet normalization)
        piece_type_transform = transforms.Compose([
            transforms.Resize(100),  # Balanced model uses 100x100
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
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
                "piece_type": "Balanced EfficientNet-B0 (Marshall + Grey Background)"
            }
        }
        
        start_time = time.time()
        
        # Process each square using exact working logic
        for rank in range(8):
            for file in range(8):
                square_img = extract_square(warped_board, rank, file)
                debug_info["squares_processed"] += 1
                
                # Occupancy detection using exact working method
                input_tensor = preprocess_square_for_occupancy(square_img).unsqueeze(0)
                with torch.no_grad():
                    occupancy_output = occupancy_model(input_tensor)
                    probs = torch.softmax(occupancy_output, dim=1)
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
                
                is_occupied = prediction == 1 and confidence > 0.3  # 1 = occupied, 0 = empty
                occupancy_list[rank * 8 + file] = is_occupied
                
                if is_occupied:
                    debug_info["occupied_squares"] += 1
                    
                    # Color classification for occupied squares
                    predicted_color = "unknown"
                    predicted_piece_type = "unknown"
                    color_confidence = 0.0
                    piece_confidence = 0.0
                    
                    if color_model and piece_type_model:
                        try:
                            # Convert BGR to RGB for PIL (square_img is in BGR from OpenCV)
                            square_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
                            
                            # Color classification (use original preprocessing)
                            input_tensor_color = color_transform(Image.fromarray(square_rgb)).unsqueeze(0)
                            with torch.no_grad():
                                color_output = color_model(input_tensor_color)
                                color_probs = torch.softmax(color_output, dim=1)
                                color_confidence = torch.max(color_probs).item()
                                predicted_color_idx = torch.argmax(color_output, dim=1).item()
                                predicted_color = COLOR_LABELS[predicted_color_idx]
                            
                            # Piece type classification (use same RGB converted square)
                            input_tensor_piece_type = piece_type_transform(Image.fromarray(square_rgb)).unsqueeze(0)
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
                            
                            # Use predictions for ALL occupied squares (no confidence threshold)
                            # If occupancy model says it's occupied, predict the piece regardless of confidence
                            if True:  # Always predict piece for occupied squares
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
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
    print("🚀 Starting Marshall Improved API on port", port)
    print("📍 This uses improved Marshall models for better accuracy")
    print("🔧 Same interface as production API")
    uvicorn.run(app, host="0.0.0.0", port=port)
