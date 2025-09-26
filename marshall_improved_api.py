#!/usr/bin/env python3
"""
Marshall Improved API
Uses the improved models trained on Marshall data without affecting current working models
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
import torch
import torch.nn as nn
from torchvision import models, transforms
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model definitions (same as training pipeline)
class CornerDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)  # 4 corners * 2 coordinates
        )
    
    def forward(self, x):
        features = self.backbone(x)
        corners = self.regressor(features)
        return corners

class OccupancyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 2)  # occupied/empty
    
    def forward(self, x):
        return self.backbone(x)

class ColorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier[1] = nn.Linear(1280, 2)  # white/black
    
    def forward(self, x):
        return self.backbone(x)

class PieceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Linear(1280, 6)  # 6 piece types
    
    def forward(self, x):
        return self.backbone(x)

# Global model instances
corner_model = None
occupancy_model = None
color_model = None
piece_model = None

# FastAPI app
app = FastAPI(title="Marshall Improved Chess API", version="2.0.0")

# Labels
COLOR_LABELS = {0: "white", 1: "black"}
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}

def preprocess_image_for_corner_detection(image):
    """Preprocess image for corner detection"""
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize
    image = image.astype(np.float32) / 255.0
    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def preprocess_square_for_occupancy(square):
    """Preprocess square for occupancy detection"""
    # Resize to 100x100
    square = cv2.resize(square, (100, 100))
    # Convert BGR to RGB
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    # Normalize
    square = square.astype(np.float32) / 255.0
    # Convert to tensor
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def preprocess_square_for_color(square):
    """Preprocess square for color classification"""
    # Resize to 224x224
    square = cv2.resize(square, (224, 224))
    # Convert BGR to RGB
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    # Normalize
    square = square.astype(np.float32) / 255.0
    # Convert to tensor
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def preprocess_square_for_piece(square):
    """Preprocess square for piece classification"""
    # Resize to 224x224
    square = cv2.resize(square, (224, 224))
    # Convert BGR to RGB
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    # Normalize
    square = square.astype(np.float32) / 255.0
    # Convert to tensor
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def warp_board(image, corners):
    """Warp image to get a square chessboard"""
    try:
        # Convert corners to numpy array
        src_points = np.array(corners, dtype=np.float32)
        
        # Define destination points for a square board
        size = 400  # 400x400 pixel board
        dst_points = np.array([
            [0, 0],
            [size, 0],
            [size, size],
            [0, size]
        ], dtype=np.float32)
        
        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp image
        warped = cv2.warpPerspective(image, matrix, (size, size))
        
        return warped
    except Exception as e:
        logger.error(f"Error warping board: {e}")
        return None

def extract_square(warped_board, rank, file):
    """Extract a single square from the warped board"""
    square_size = warped_board.shape[0] // 8
    
    y1 = rank * square_size
    y2 = (rank + 1) * square_size
    x1 = file * square_size
    x2 = (file + 1) * square_size
    
    # Extract square
    square = warped_board[y1:y2, x1:x2]
    
    return square

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Marshall Improved API...")
    logger.info("📍 Running on port 8006 (separate from all other APIs)")
    
    global corner_model, occupancy_model, color_model, piece_model
    
    models_dir = Path("models_marshall_improved")
    
    # Load corner detection model
    logger.info("Loading Marshall corner detection model...")
    corner_model = CornerDetectionModel()
    corner_model_path = models_dir / "corner_detection_marshall.pt"
    if corner_model_path.exists():
        corner_model.load_state_dict(torch.load(str(corner_model_path), map_location='cpu'))
        corner_model.eval()
        logger.info("✅ Marshall corner detection model loaded")
    else:
        logger.error(f"Corner detection model not found at {corner_model_path}")
        raise RuntimeError("Corner detection model not found")
    
    # Load occupancy model
    logger.info("Loading Marshall occupancy model...")
    occupancy_model = OccupancyModel()
    occupancy_model_path = models_dir / "occupancy_marshall.pt"
    if occupancy_model_path.exists():
        occupancy_model.load_state_dict(torch.load(str(occupancy_model_path), map_location='cpu'))
        occupancy_model.eval()
        logger.info("✅ Marshall occupancy model loaded")
    else:
        logger.error(f"Occupancy model not found at {occupancy_model_path}")
        raise RuntimeError("Occupancy model not found")
    
    # Load color model
    logger.info("Loading Marshall color model...")
    color_model = ColorModel()
    color_model_path = models_dir / "color_classification_marshall.pt"
    if color_model_path.exists():
        color_model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        color_model.eval()
        logger.info("✅ Marshall color model loaded")
    else:
        logger.error(f"Color model not found at {color_model_path}")
        raise RuntimeError("Color model not found")
    
    # Load piece model
    logger.info("Loading Marshall piece model...")
    piece_model = PieceModel()
    piece_model_path = models_dir / "piece_classification_marshall.pt"
    if piece_model_path.exists():
        piece_model.load_state_dict(torch.load(str(piece_model_path), map_location='cpu'))
        piece_model.eval()
        logger.info("✅ Marshall piece model loaded")
    else:
        logger.error(f"Piece model not found at {piece_model_path}")
        raise RuntimeError("Piece model not found")
    
    logger.info("🎉 Marshall Improved API startup completed successfully")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api": "marshall_improved", "port": 8006}

@app.post("/detect_corners")
async def detect_corners_marshall(file: UploadFile = File(...)):
    """Detect chessboard corners using Marshall improved model"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Preprocess image
        image_tensor = preprocess_image_for_corner_detection(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = corner_model(image_tensor)
            corners_normalized = corners_normalized.squeeze().cpu().numpy()
        
        # Convert normalized corners back to image coordinates
        h, w = image.shape[:2]
        corners = corners_normalized * np.array([w, h])
        corners = corners.reshape(4, 2).astype(int).tolist()
        
        return {
            "corners": corners,
            "confidence": 0.95,  # Placeholder confidence
            "model": "marshall_improved"
        }
        
    except Exception as e:
        logger.error(f"Error in corner detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_position")
async def analyze_position_marshall(
    file: UploadFile = File(...),
    corners: str = Form(...),
    fen: str = Form("")
):
    """Analyze chess position using Marshall improved models"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Parse corners
        corners_list = json.loads(corners)
        if len(corners_list) != 4:
            raise HTTPException(status_code=400, detail="Invalid corners format")
        
        # Warp board
        warped_board = warp_board(image, corners_list)
        if warped_board is None:
            raise HTTPException(status_code=400, detail="Could not warp board")
        
        # Analyze each square
        board_state = []
        occupancy_count = 0
        
        for rank in range(8):
            rank_state = []
            for file in range(8):
                # Extract square
                square = extract_square(warped_board, rank, file)
                
                # Occupancy detection
                square_tensor = preprocess_square_for_occupancy(square)
                square_tensor = square_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    occupancy_output = occupancy_model(square_tensor)
                    occupancy_prob = torch.softmax(occupancy_output, dim=1)
                    is_occupied = occupancy_prob[0][1].item() > 0.5
                
                if is_occupied:
                    occupancy_count += 1
                    
                    # Color classification
                    color_tensor = preprocess_square_for_color(square)
                    color_tensor = color_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        color_output = color_model(color_tensor)
                        color_prob = torch.softmax(color_output, dim=1)
                        color_pred = torch.argmax(color_prob, dim=1).item()
                        color_confidence = color_prob[0][color_pred].item()
                    
                    # Piece classification
                    piece_tensor = preprocess_square_for_piece(square)
                    piece_tensor = piece_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        piece_output = piece_model(piece_tensor)
                        piece_prob = torch.softmax(piece_output, dim=1)
                        piece_pred = torch.argmax(piece_prob, dim=1).item()
                        piece_confidence = piece_prob[0][piece_pred].item()
                    
                    # Combine color and piece
                    color = COLOR_LABELS[color_pred]
                    piece_type = PIECE_TYPE_LABELS[piece_pred]
                    piece = f"{color}_{piece_type}"
                    
                    rank_state.append({
                        "piece": piece,
                        "color": color,
                        "type": piece_type,
                        "confidence": {
                            "color": color_confidence,
                            "piece": piece_confidence,
                            "occupancy": occupancy_prob[0][1].item()
                        }
                    })
                else:
                    rank_state.append({
                        "piece": None,
                        "color": None,
                        "type": None,
                        "confidence": {
                            "occupancy": occupancy_prob[0][0].item()
                        }
                    })
            
            board_state.append(rank_state)
        
        # Generate FEN
        fen = generate_fen_from_board_state(board_state)
        
        return {
            "board_state": board_state,
            "fen": fen,
            "occupancy_count": occupancy_count,
            "model": "marshall_improved",
            "processing_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in position analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_fen_from_board_state(board_state):
    """Generate FEN from board state"""
    fen_parts = []
    
    for rank in board_state:
        rank_str = ""
        empty_count = 0
        
        for square in rank:
            if square["piece"] is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                
                # Convert piece to FEN notation
                piece = square["piece"]
                if piece.startswith("white_"):
                    piece_char = piece.split("_")[1][0].upper()
                else:  # black
                    piece_char = piece.split("_")[1][0].lower()
                
                rank_str += piece_char
        
        if empty_count > 0:
            rank_str += str(empty_count)
        
        fen_parts.append(rank_str)
    
    # Add other FEN components (simplified)
    fen = "/".join(fen_parts) + " w - - 0 1"
    
    return fen

@app.post("/visualize_corners")
async def visualize_corners_marshall(file: UploadFile = File(...)):
    """Visualize detected corners on the image"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Detect corners
        corners_response = await detect_corners_marshall(file)
        corners = corners_response["corners"]
        
        # Draw corners
        vis_image = image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = corner
            cv2.circle(vis_image, (x, y), 15, color, -1)
            cv2.circle(vis_image, (x, y), 20, (255, 255, 255), 3)
            cv2.putText(vis_image, f'{label}', (x-20, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw quadrilateral
        corners_np = np.array(corners, dtype=np.int32)
        cv2.polylines(vis_image, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        
        # Encode image
        _, buffer = cv2.imencode('.jpg', vis_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": image_base64,
            "corners": corners,
            "model": "marshall_improved"
        }
        
    except Exception as e:
        logger.error(f"Error in corner visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

