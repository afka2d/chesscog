from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import base64
import io
import json
import logging
import traceback
import time
import numpy as np
import cv2
import chess
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from torchvision import transforms

from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image, crop_square as crop_occupancy_square
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from chesscog.piece_classifier.create_dataset import crop_square as crop_piece_square
from chesscog.core import sort_corner_points

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Position Scanner API",
    description="API for recognizing chess positions from images with manual corner coordinates",
    version="1.0.0"
)

# Add CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
cfg = None
recognizer = None
custom_piece_model = None
custom_piece_transforms = None

class CustomChessRecognizer(ChessRecognizer):
    """
    Custom chess recognizer that uses the improved ResNet_uniform model.
    """
    
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.custom_piece_model = None
        self.custom_piece_transforms = None
        self._load_custom_piece_model()
    
    def _load_custom_piece_model(self):
        """Load the custom piece classification model."""
        try:
            # Load the improved ResNet_uniform model
            model_path = "runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt"
            if os.path.exists(model_path):
                logger.info(f"Loading custom piece model from {model_path}")
                self.custom_piece_model = torch.load(model_path, map_location='cpu', weights_only=False)
                self.custom_piece_model.eval()
                
                # Define transforms for the custom model
                self.custom_piece_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((100, 100)),  # Match the training configuration
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                logger.info("Custom piece model loaded successfully")
            else:
                logger.warning(f"Custom piece model not found at {model_path}, using default model")
        except Exception as e:
            logger.error(f"Failed to load custom piece model: {e}")
            logger.warning("Falling back to default piece model")
    
    def _classify_pieces(self, img, turn, corners, occupancy):
        """
        Classify pieces using the custom model if available.
        """
        if self.custom_piece_model is None:
            # Fall back to parent method
            return super()._classify_pieces(img, turn, corners, occupancy)
        
        try:
            logger.info("Using custom piece classification model")
            
            # Warp the chessboard
            warped = warp_chessboard_image(img, corners)
            
            # Get piece classes from the custom model
            piece_classes = [
                'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
                'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
            ]
            
            pieces = np.full((8, 8), None, dtype=object)
            
            for rank in range(8):
                for file in range(8):
                    if occupancy[rank, file]:
                        # Crop the square
                        square_img = crop_piece_square(warped, rank, file)
                        
                        # Preprocess for the custom model
                        square_tensor = self.custom_piece_transforms(square_img).unsqueeze(0)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = self.custom_piece_model(square_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                        
                        # Only use prediction if confidence is high enough
                        if confidence > 0.3:  # Adjustable threshold
                            piece_name = piece_classes[predicted_class]
                            
                            # Parse piece name to get color and type
                            if piece_name.startswith('white_'):
                                color = chess.WHITE
                                piece_type = piece_name[6:]  # Remove 'white_' prefix
                            else:
                                color = chess.BLACK
                                piece_type = piece_name[6:]  # Remove 'black_' prefix
                            
                            # Convert piece type to chess piece
                            piece_map = {
                                'pawn': chess.PAWN,
                                'rook': chess.ROOK,
                                'knight': chess.KNIGHT,
                                'bishop': chess.BISHOP,
                                'queen': chess.QUEEN,
                                'king': chess.KING
                            }
                            
                            if piece_type in piece_map:
                                pieces[rank, file] = chess.Piece(piece_map[piece_type], color)
                                logger.debug(f"Square {rank},{file}: {piece_name} (conf: {confidence:.3f})")
                            else:
                                logger.warning(f"Unknown piece type: {piece_type}")
                        else:
                            logger.debug(f"Square {rank},{file}: Low confidence ({confidence:.3f}), skipping")
            
            return pieces
            
        except Exception as e:
            logger.error(f"Custom piece classification failed: {e}")
            logger.warning("Falling back to default piece classification")
            return super()._classify_pieces(img, turn, corners, occupancy)

def encode_image(image, max_width=800, max_height=600):
    """Encode image to base64 string with size constraints."""
    try:
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            h, w = image.shape[:2]
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Convert to PIL Image for better format handling
            pil_image = Image.fromarray(image)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        return None
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return None

def create_board_focus_debug_image(img, corners):
    """Create a debug image showing the board focus area."""
    try:
        # Create a copy of the image
        debug_img = img.copy()
        
        # Create a mask for the board area
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        corners_int = corners.astype(np.int32)
        cv2.fillPoly(mask, [corners_int], 255)
        
        # Apply Gaussian blur to non-board areas
        blurred = cv2.GaussianBlur(img, (51, 51), 0)
        
        # Combine original board area with blurred background
        debug_img = np.where(mask[:, :, np.newaxis] == 255, img, blurred)
        
        # Draw corner points
        for i, corner in enumerate(corners_int):
            cv2.circle(debug_img, tuple(corner), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, str(i+1), tuple(corner), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return debug_img
    except Exception as e:
        logger.error(f"Failed to create board focus debug image: {e}")
        return img

@app.on_event("startup")
async def startup_event():
    """Initialize models and configurations on startup."""
    global cfg, recognizer, custom_piece_model, custom_piece_transforms
    
    try:
        logger.info("Starting up Chess Position Scanner API...")
        
        # Load configuration
        logger.info("Loading configuration...")
        cfg = CN.load_yaml_with_base('config/recognition.yaml')
        logger.info("Configuration loaded successfully")
        
        # Initialize the custom recognizer
        logger.info("Initializing custom chess recognizer...")
        recognizer = CustomChessRecognizer(cfg)
        logger.info("Custom chess recognizer initialized successfully")
        
        # Test model loading
        logger.info("Testing model loading...")
        if recognizer.custom_piece_model is not None:
            logger.info("Custom piece model loaded successfully")
        else:
            logger.warning("Custom piece model not available, using default")
        
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise RuntimeError(f"Startup failed: {e}")

@app.post("/recognize_chess_position_with_corners")
async def recognize_chess_position_with_corners(
    image: UploadFile = File(...),
    corners: str = Form(...),  # JSON string of corner coordinates
    color: str = "white",
    debug_image_width: int = 800,
    debug_image_height: int = 600
):
    """
    Recognize chess position from uploaded image using manually corrected corner coordinates.
    
    Args:
        image: Chess board image (JPEG or PNG)
        corners: JSON string of corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        color: Color to play as ("white" or "black")
        debug_image_width: Maximum width for debug images
        debug_image_height: Maximum height for debug images
    
    Returns:
        JSON response with FEN, ASCII board, Lichess URL, and debug images
    """
    try:
        # Validate input
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Parse corner coordinates
        try:
            import json
            corner_coords = json.loads(corners)
            if not isinstance(corner_coords, list) or len(corner_coords) != 4:
                raise ValueError("Corners must be a list of 4 coordinate pairs")
            corners_array = np.array(corner_coords, dtype=np.float32)
            if corners_array.shape != (4, 2):
                raise ValueError("Each corner must have 2 coordinates (x, y)")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid corner coordinates: {str(e)}")
        
        # Read and validate image
        img_bytes = await image.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Validate color parameter
        if color.lower() not in ["white", "black"]:
            raise HTTPException(status_code=400, detail="Color must be 'white' or 'black'")
        
        turn = chess.WHITE if color.lower() == "white" else chess.BLACK
        
        logger.info(f"Processing image with manual corners: {image.filename}")
        logger.info(f"Corner coordinates: {corner_coords}")
        
        # Use the recognizer's predict_with_debug method but skip corner detection
        try:
            # Create debug images dictionary starting with the resized image
            debug_images = {}
            
            # Resize image for processing
            resized_img = cv2.resize(img, (800, 600))
            debug_images['resized'] = resized_img.copy()
            
            # Use the provided corners directly
            logger.info("Using manually provided corner coordinates")
            
            # Warp the chessboard using the provided corners
            from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image
            warped_board = warp_chessboard_image(img, corners_array)
            debug_images['warped_board'] = warped_board.copy()
            
            # Create board focus debug image (clear board, blurred outside)
            board_focus_img = create_board_focus_debug_image(img, corners_array)
            debug_images['board_focus'] = board_focus_img
            
            # Classify occupancy
            logger.info("Classifying occupancy...")
            occupancy = recognizer._classify_occupancy(img, turn, corners_array)
            debug_images['occupancy_map'] = recognizer._visualize_occupancy_map(warped_board, occupancy, turn)
            
            # Classify pieces
            logger.info("Classifying pieces...")
            pieces = recognizer._classify_pieces(img, turn, corners_array, occupancy)
            debug_images['piece_map'] = recognizer._visualize_piece_map(warped_board, pieces, occupancy, turn)
            
            # Create the chess board
            logger.info("Creating chess board...")
            board = chess.Board()
            board.clear()
            
            # Place pieces on the board
            for rank in range(8):
                for file in range(8):
                    if occupancy[rank, file] and pieces[rank, file] is not None:
                        square = chess.square(file, 7 - rank)  # Convert to chess square (a1 is bottom-left)
                        board.set_piece_at(square, pieces[rank, file])
            
            # Generate results
            fen = board.fen()
            ascii_board = str(board)
            lichess_url = f"https://lichess.org/editor/{fen}?color={color}"
            legal = board.is_valid()
            
            # Count pieces found
            piece_count = len(board.piece_map())
            
            # Generate human-readable description
            position_description = generate_position_description(board, color)
            
            # Convert debug images to base64
            debug_images_base64 = {}
            for key, img in debug_images.items():
                encoded = encode_image(img, debug_image_width, debug_image_height)
                if encoded:
                    debug_images_base64[key] = encoded
            
            # Get image info
            height, width = img.shape[:2]
            image_info = {
                "filename": image.filename,
                "content_type": image.content_type,
                "size_bytes": len(img_bytes),
                "shape": [height, width, 3]
            }
            
            # Create debug info
            debug_info = {
                "corner_detection": "Manual (provided by user)",
                "board_warping": "Completed",
                "position_detection": "Completed",
                "visualization": "Completed",
                "description_generation": "Completed"
            }
            
            logger.info(f"Recognition successful: FEN={fen}, Legal={legal}, Pieces={piece_count}")
            
            return JSONResponse(
                content={
                    "fen": fen,
                    "ascii": ascii_board,
                    "lichess_url": lichess_url,
                    "legal_position": legal,
                    "position_description": position_description,
                    "debug_images": debug_images_base64,
                    "corners": corner_coords,
                    "processing_time": time.time(),
                    "image_info": image_info,
                    "debug_info": debug_info
                }
            )
            
        except Exception as e:
            logger.error(f"Manual corner recognition failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, 
                detail=f"Manual corner recognition failed: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

def generate_position_description(board, color):
    """Generate a human-readable description of the chess position."""
    try:
        description_parts = []
        
        # Count pieces by type and color
        piece_counts = {}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_name = piece.symbol().upper()
                piece_color = "white" if piece.color == chess.WHITE else "black"
                key = f"{piece_color}_{piece_name}"
                piece_counts[key] = piece_counts.get(key, 0) + 1
        
        # Generate description
        if color == "white":
            description_parts.append("You are playing as White.")
        else:
            description_parts.append("You are playing as Black.")
        
        # Describe the position
        if len(board.piece_map()) == 0:
            description_parts.append("The board is completely empty.")
        else:
            # Count total pieces
            total_pieces = len(board.piece_map())
            description_parts.append(f"There are {total_pieces} pieces on the board.")
            
            # Describe key pieces
            white_king = None
            black_king = None
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.KING:
                    if piece.color == chess.WHITE:
                        white_king = chess.square_name(square)
                    else:
                        black_king = chess.square_name(square)
            
            if white_king:
                description_parts.append(f"White king is on {white_king}.")
            if black_king:
                description_parts.append(f"Black king is on {black_king}.")
            
            # Describe material advantage
            white_material = sum(1 for square in chess.SQUARES 
                               if board.piece_at(square) and board.piece_at(square).color == chess.WHITE)
            black_material = sum(1 for square in chess.SQUARES 
                               if board.piece_at(square) and board.piece_at(square).color == chess.BLACK)
            
            if white_material > black_material:
                description_parts.append(f"White has material advantage ({white_material} vs {black_material} pieces).")
            elif black_material > white_material:
                description_parts.append(f"Black has material advantage ({black_material} vs {white_material} pieces).")
            else:
                description_parts.append("Material is equal.")
        
        # Check for special conditions
        if board.is_check():
            description_parts.append("The position is in check.")
        if board.is_checkmate():
            description_parts.append("This is checkmate!")
        elif board.is_stalemate():
            description_parts.append("This is stalemate.")
        
        return " ".join(description_parts)
        
    except Exception as e:
        logger.error(f"Failed to generate position description: {e}")
        return "Position description could not be generated."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        log_level="info",
        access_log=True
    )
