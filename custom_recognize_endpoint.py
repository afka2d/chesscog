#!/usr/bin/env python3
"""
Custom recognition endpoint that can use manual corner annotations
"""

import cv2
import numpy as np
import chess
import json
from pathlib import Path
from chesscog.recognition import ChessRecognizer
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image

def load_annotation(image_name):
    """Load corner annotation for a given image."""
    annotation_path = f"grey_background_dataset/annotations/test/{image_name}.json"
    if Path(annotation_path).exists():
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            return data.get('corners', None)
    return None

def recognize_with_manual_corners(image_path, color="white"):
    """Recognize chess position using manual corner annotations."""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get image name
    image_name = Path(image_path).stem
    
    # Load manual corners
    corners = load_annotation(image_name)
    if corners is None:
        raise ValueError(f"No manual annotation found for {image_name}")
    
    # Convert corners to numpy array
    corners_array = np.array(corners, dtype=np.float32)
    
    # Warp the chessboard using manual corners
    warped_board = warp_chessboard_image(img, corners_array)
    
    # Initialize recognizer (this will use the updated models)
    recognizer = ChessRecognizer(Path("models"))
    
    # Use the warped board for recognition
    # Note: This is a simplified approach - in practice, you'd need to modify
    # the recognizer to accept pre-warped images or manual corners
    
    print(f"Manual corners for {image_name}: {corners}")
    print(f"Warped board shape: {warped_board.shape}")
    
    return {
        "image_name": image_name,
        "manual_corners": corners,
        "warped_shape": warped_board.shape,
        "status": "Manual corners loaded successfully"
    }

if __name__ == "__main__":
    # Test with the problematic image
    result = recognize_with_manual_corners("grey_background_dataset/images/test/IMG_4752.JPG")
    print(json.dumps(result, indent=2)) 