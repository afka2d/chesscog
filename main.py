from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import base64
import io
import logging
import traceback
import time
import numpy as np
import cv2
import chess
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image, crop_square as crop_occupancy_square
from chesscog.piece_classifier.create_dataset import crop_square as crop_piece_square
from chesscog.core import sort_corner_points

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chess Position Scanner API",
    description="API for recognizing chess positions from images",
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

def load_models():
    """Load the chess recognition models."""
    global cfg, recognizer
    try:
        logger.info("Loading corner detection configuration...")
        cfg = CN.load_yaml_with_base("config/corner_detection.yaml")
        
        logger.info("Loading chess recognizer...")
        recognizer = ChessRecognizer(Path("models"))
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def encode_image(img, max_width=800, max_height=600):
    """Convert a numpy array image to base64 string with optional resizing."""
    try:
        if img is None:
            return None
        
        # Ensure image is in the correct format
        if len(img.shape) == 3 and img.shape[2] == 3:
            # BGR to RGB conversion for better compatibility
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Resize image if it's too large for mobile display
        height, width = img_rgb.shape[:2]
        if width > max_width or height > max_height:
            # Calculate new dimensions maintaining aspect ratio
            aspect_ratio = width / height
            if width > height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            # Resize the image
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized debug image from {width}x{height} to {new_width}x{new_height}")
            
        # Encode as PNG for better quality
        _, buffer = cv2.imencode('.png', img_rgb)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return None

def create_chess_board_visualization(board, max_width=400, max_height=400):
    """Create a visual representation of the detected chess board."""
    try:
        # Create a chess board image
        board_size = 400
        square_size = board_size // 8
        
        # Create image with white background
        img = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255
        
        # Draw chess board pattern
        for rank in range(8):
            for file in range(8):
                if (rank + file) % 2 == 0:
                    color = (240, 240, 240)  # Light square
                else:
                    color = (120, 120, 120)  # Dark square
                
                x1 = file * square_size
                y1 = rank * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Draw pieces
        piece_symbols = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Convert to board coordinates (flip rank for display)
                display_rank = 7 - rank
                display_file = file
                
                # Calculate position
                x = display_file * square_size + square_size // 2
                y = display_rank * square_size + square_size // 2
                
                # Get piece symbol
                piece_char = piece.symbol()
                if piece.color == chess.WHITE:
                    piece_char = piece_char.upper()
                else:
                    piece_char = piece_char.lower()
                
                symbol = piece_symbols.get(piece_char, piece_char)
                
                # Convert to PIL for text rendering
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)
                
                # Try to use a font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 30)
                except:
                    font = ImageFont.load_default()
                
                # Draw piece
                color = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                draw.text((x-10, y-15), symbol, fill=color, font=font)
                
                img = np.array(pil_img)
        
        # Resize if needed
        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            if width > height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return encode_image(img)
    except Exception as e:
        logger.error(f"Failed to create chess board visualization: {e}")
        return None

def create_square_grid_visualization(warped_img, occupancy_results, piece_results, max_width=600, max_height=600):
    """Create a visualization showing all 64 squares with their classification results."""
    try:
        # Create a grid of squares
        grid_size = 8
        square_size = 60
        margin = 10
        total_size = grid_size * square_size + (grid_size + 1) * margin
        
        # Create image with white background
        img = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Convert to chess square
                
                # Calculate position
                x = file * square_size + (file + 1) * margin
                y = rank * square_size + (rank + 1) * margin
                
                # Determine square color
                if (rank + file) % 2 == 0:
                    color = (240, 240, 240)  # Light square
                else:
                    color = (120, 120, 120)  # Dark square
                
                # Draw square
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), color, -1)
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), (0, 0, 0), 1)
                
                # Add occupancy result
                if occupancy_results[square]:
                    cv2.circle(img, (x + square_size//2, y + square_size//2), 5, (0, 255, 0), -1)
                
                # Add piece result if available
                if piece_results and piece_results[square]:
                    piece = piece_results[square]
                    piece_char = piece.symbol()
                    if piece.color == chess.WHITE:
                        piece_char = piece_char.upper()
                    else:
                        piece_char = piece_char.lower()
                    
                    # Convert to PIL for text rendering
                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    color = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                    draw.text((x + 5, y + 5), piece_char, fill=color, font=font)
                    
                    img = np.array(pil_img)
        
        # Resize if needed
        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            if width > height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return encode_image(img)
    except Exception as e:
        logger.error(f"Failed to create square grid visualization: {e}")
        return None

def create_board_focus_debug_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Create a debug image showing the original image with the board area clearly visible
    and everything outside the detected corners blurred.
    
    Args:
        img: Original image
        corners: Corner coordinates as numpy array of shape (4, 2)
    
    Returns:
        Debug image with board area clear and outside area blurred
    """
    try:
        # Create a copy of the original image
        debug_img = img.copy()
        
        # Sort corners to ensure consistent order
        from chesscog.core import sort_corner_points
        sorted_corners = sort_corner_points(corners)
        
        # Create a mask for the board area
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Convert corners to integer coordinates for the mask
        corner_points = sorted_corners.astype(np.int32)
        
        # Fill the board area with white (255)
        cv2.fillPoly(mask, [corner_points], 255)
        
        # Create a blurred version of the entire image
        blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
        
        # Combine the original image (board area) with blurred image (outside area)
        # Where mask is 255 (board area), use original image
        # Where mask is 0 (outside area), use blurred image
        debug_img = np.where(mask[:, :, np.newaxis] == 255, img, blurred_img)
        
        # Draw corner points and board outline for clarity
        for i, corner in enumerate(sorted_corners):
            x, y = int(corner[0]), int(corner[1])
            # Draw corner points
            cv2.circle(debug_img, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(debug_img, (x, y), 8, (0, 0, 0), 2)    # Black outline
            # Add corner labels
            cv2.putText(debug_img, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw board outline
        cv2.polylines(debug_img, [corner_points], True, (0, 255, 0), 3)
        
        return debug_img.astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Failed to create board focus debug image: {e}")
        return img  # Return original image if processing fails

def generate_position_description(board, color_perspective="white"):
    """
    Generate a human-readable description of the chess position.
    
    Args:
        board: chess.Board object
        color_perspective: "white" or "black" perspective
    
    Returns:
        str: Human-readable description of the position
    """
    piece_map = board.piece_map()
    
    if not piece_map:
        return "The board is empty with no pieces."
    
    # Group pieces by color and type
    white_pieces = {}
    black_pieces = {}
    
    for square, piece in piece_map.items():
        square_name = chess.square_name(square)
        piece_symbol = piece.symbol()
        piece_name = {
            'P': 'Pawn', 'R': 'Rook', 'N': 'Knight', 
            'B': 'Bishop', 'Q': 'Queen', 'K': 'King'
        }.get(piece_symbol.upper(), piece_symbol)
        
        if piece.color:  # White piece
            if piece_name not in white_pieces:
                white_pieces[piece_name] = []
            white_pieces[piece_name].append(square_name)
        else:  # Black piece
            if piece_name not in black_pieces:
                black_pieces[piece_name] = []
            black_pieces[piece_name].append(square_name)
    
    # Build description
    description_parts = []
    
    # White pieces
    if white_pieces:
        white_desc = []
        for piece_name, squares in white_pieces.items():
            if len(squares) == 1:
                white_desc.append(f"White {piece_name} on {squares[0]}")
            else:
                white_desc.append(f"White {piece_name}s on {', '.join(squares)}")
        description_parts.append("White pieces: " + "; ".join(white_desc))
    
    # Black pieces
    if black_pieces:
        black_desc = []
        for piece_name, squares in black_pieces.items():
            if len(squares) == 1:
                black_desc.append(f"Black {piece_name} on {squares[0]}")
            else:
                black_desc.append(f"Black {piece_name}s on {', '.join(squares)}")
        description_parts.append("Black pieces: " + "; ".join(black_desc))
    
    # Add turn information
    turn = "White" if board.turn else "Black"
    description_parts.append(f"{turn} to move")
    
    # Add castling rights
    castling_rights = []
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights.append("White kingside")
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights.append("White queenside")
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights.append("Black kingside")
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights.append("Black queenside")
    
    if castling_rights:
        description_parts.append(f"Castling available: {', '.join(castling_rights)}")
    
    # Add en passant if available
    if board.ep_square:
        ep_square = chess.square_name(board.ep_square)
        description_parts.append(f"En passant available on {ep_square}")
    
    return ". ".join(description_parts) + "."

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Chess Position Scanner API...")
    if not load_models():
        logger.error("Failed to load models during startup")
        raise RuntimeError("Failed to load models")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Chess Position Scanner API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    models_loaded = cfg is not None and recognizer is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "timestamp": time.time()
    }

@app.post("/recognize_chess_position")
async def recognize_chess_position(
    image: UploadFile = File(...), 
    color: str = "white",
    debug_image_width: int = 800,
    debug_image_height: int = 600
):
    """
    Recognize chess position from uploaded image.
    
    Args:
        image: Chess board image (JPEG or PNG)
        color: Color to play as ("white" or "black")
        debug_image_width: Maximum width for debug images
        debug_image_height: Maximum height for debug images
    
    Returns:
        JSON with FEN notation, ASCII board, Lichess URL, legal position status, and debug images
    """
    if not cfg or not recognizer:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported image type. Use JPEG or PNG."
        )
    
    try:
        # Read and decode image
        img_bytes = await image.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"Processing image: {image.filename}, shape: {img.shape}")
        
        # Validate color parameter
        if color not in ["white", "black"]:
            color = "white"  # Default to white
        
        chess_color = chess.WHITE if color == "white" else chess.BLACK
        
        # Perform recognition with debug images
        logger.info("Performing chess recognition with debug images...")
        board, corners, debug_images = recognizer.predict_with_debug(img, chess_color)
        
        # Save debug images to disk
        import os
        debug_output_dir = "debug_outputs"
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # Save each debug image (overwriting previous versions)
        debug_image_paths = {}
        
        for key, img in debug_images.items():
            if isinstance(img, np.ndarray):
                filename = f"{key}.png"
                filepath = os.path.join(debug_output_dir, filename)
                cv2.imwrite(filepath, img)
                debug_image_paths[key] = filepath
                logger.info(f"Saved debug image: {filepath}")
        
        # Convert debug images to base64
        debug_images_base64 = {}
        for key, img in debug_images.items():
            encoded = encode_image(img, debug_image_width, debug_image_height)
            if encoded:
                debug_images_base64[key] = encoded
        
        # Generate results
        fen = board.fen()
        ascii_board = str(board)
        lichess_url = f"https://lichess.org/editor/{fen}?color={color}"
        legal = board.is_valid()
        
        # Generate human-readable description
        position_description = generate_position_description(board, color)
        
        logger.info(f"Recognition successful: FEN={fen}, Legal={legal}")
        
        return JSONResponse(
            content={
                "fen": fen,
                "ascii": ascii_board,
                "lichess_url": lichess_url,
                "legal_position": legal,
                "position_description": position_description,
                "debug_images": debug_images_base64,
                "debug_image_paths": debug_image_paths,
                "corners": corners.tolist() if corners is not None else None,
                "processing_time": time.time(),
                "image_info": {
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "size_bytes": len(img_bytes),
                    "shape": img.shape
                },
                "debug_info": {
                    "corner_detection": "Completed",
                    "board_warping": "Completed",
                    "position_detection": "Completed",
                    "visualization": "Completed",
                    "description_generation": "Completed"
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Recognition failed: {str(e)}"
        )

@app.post("/detect_corners")
async def detect_corners(image: UploadFile = File(...)):
    """
    Detect chess board corners from uploaded image.
    
    Args:
        image: Chess board image (JPEG or PNG)
    
    Returns:
        JSON with detected corners and debug images
    """
    if not cfg:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported image type. Use JPEG or PNG."
        )
    
    try:
        # Read and decode image
        img_bytes = await image.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"Detecting corners for image: {image.filename}")
        
        # Perform corner detection
        corners, debug_images = find_corners(cfg, img)
        
        # Convert debug images to base64
        debug_images_base64 = {}
        for key, img in debug_images.items():
            encoded = encode_image(img, 800, 600)
            if encoded:
                debug_images_base64[key] = encoded
        
        # Convert corners to list format for JSON serialization
        corners_list = corners.tolist() if corners is not None else None
        
        logger.info(f"Corner detection successful: {len(corners_list) if corners_list else 0} corners")
        
        return JSONResponse(
            content={
                "corners": corners_list,
                "message": "Successfully detected chessboard corners",
                "debug_images": debug_images_base64,
                "processing_time": time.time(),
                "image_info": {
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "size_bytes": len(img_bytes),
                    "shape": img.shape
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Corner detection failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Corner detection failed: {str(e)}"
        )

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
            for square, piece in zip(recognizer._squares, pieces):
                if piece is not None:
                    board.set_piece_at(square, piece)
            
            # Set the turn
            board.turn = turn
            
            # Generate results
            fen = board.fen()
            ascii_board = str(board)
            lichess_url = f"https://lichess.org/editor/{fen}?color={color.lower()}"
            legal = board.is_valid()
            
            logger.info(f"Recognition successful: FEN={fen}, Legal={legal}")
            
        except Exception as e:
            logger.error(f"Recognition failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
        
        # Save debug images to disk
        import os
        debug_output_dir = "debug_outputs"
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # Save each debug image (overwriting previous versions)
        debug_image_paths = {}
        
        for key, img in debug_images.items():
            if isinstance(img, np.ndarray):
                filename = f"{key}.png"
                filepath = os.path.join(debug_output_dir, filename)
                cv2.imwrite(filepath, img)
                debug_image_paths[key] = filepath
                logger.info(f"Saved debug image: {filepath}")
        
        # Convert debug images to base64
        debug_images_base64 = {}
        for key, img in debug_images.items():
            encoded = encode_image(img, debug_image_width, debug_image_height)
            if encoded:
                debug_images_base64[key] = encoded
        
        return JSONResponse(
            content={
                "fen": fen,
                "ascii": ascii_board,
                "lichess_url": lichess_url,
                "legal_position": legal,
                "debug_images": debug_images_base64,
                "debug_image_paths": debug_image_paths,
                "corners": corners_array.tolist(),
                "processing_time": time.time(),
                "image_info": {
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "size_bytes": len(img_bytes),
                    "shape": img.shape
                },
                "debug_info": {
                    "corner_detection": "Skipped (manual input)",
                    "board_warping": "Completed",
                    "position_detection": "Completed",
                    "visualization": "Completed"
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recognize_chess_position_with_description")
async def recognize_chess_position_with_description(
    image: UploadFile = File(...), 
    color: str = "white",
    debug_image_width: int = 800,
    debug_image_height: int = 600
):
    """
    Recognize chess position from uploaded image and provide human-readable description.
    
    Args:
        image: Chess board image (JPEG or PNG)
        color: Color to play as ("white" or "black")
        debug_image_width: Maximum width for debug images
        debug_image_height: Maximum height for debug images
    
    Returns:
        JSON with FEN notation, ASCII board, Lichess URL, legal position status, 
        human-readable description, and debug images
    """
    if not cfg or not recognizer:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image type
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported image type. Use JPEG or PNG."
        )
    
    try:
        # Read and decode image
        img_bytes = await image.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"Processing image: {image.filename}, shape: {img.shape}")
        
        # Validate color parameter
        if color not in ["white", "black"]:
            color = "white"  # Default to white
        
        chess_color = chess.WHITE if color == "white" else chess.BLACK
        
        # Perform recognition with debug images
        logger.info("Performing chess recognition with debug images...")
        board, corners, debug_images = recognizer.predict_with_debug(img, chess_color)
        
        # Save debug images to disk
        import os
        debug_output_dir = "debug_outputs"
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # Save each debug image (overwriting previous versions)
        debug_image_paths = {}
        
        for key, img in debug_images.items():
            if isinstance(img, np.ndarray):
                filename = f"{key}.png"
                filepath = os.path.join(debug_output_dir, filename)
                cv2.imwrite(filepath, img)
                debug_image_paths[key] = filepath
                logger.info(f"Saved debug image: {filepath}")
        
        # Convert debug images to base64
        debug_images_base64 = {}
        for key, img in debug_images.items():
            encoded = encode_image(img, debug_image_width, debug_image_height)
            if encoded:
                debug_images_base64[key] = encoded
        
        # Generate results
        fen = board.fen()
        ascii_board = str(board)
        lichess_url = f"https://lichess.org/editor/{fen}?color={color}"
        legal = board.is_valid()
        
        # Generate human-readable description
        position_description = generate_position_description(board, color)
        
        logger.info(f"Recognition successful: FEN={fen}, Legal={legal}")
        
        return JSONResponse(
            content={
                "fen": fen,
                "ascii": ascii_board,
                "lichess_url": lichess_url,
                "legal_position": legal,
                "position_description": position_description,
                "debug_images": debug_images_base64,
                "debug_image_paths": debug_image_paths,
                "corners": corners.tolist() if corners is not None else None,
                "processing_time": time.time(),
                "image_info": {
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "size_bytes": len(img_bytes),
                    "shape": img.shape
                },
                "debug_info": {
                    "corner_detection": "Completed",
                    "board_warping": "Completed",
                    "position_detection": "Completed",
                    "visualization": "Completed",
                    "description_generation": "Completed"
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Recognition failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info",
        access_log=True
    )
