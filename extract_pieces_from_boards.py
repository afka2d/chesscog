#!/usr/bin/env python3
"""
Extract individual chess pieces from full board images and add them to the training dataset.
This script will process the additional 397 training images and extract pieces to enhance the dataset.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_pieces_from_board(board_image_path, output_dir, piece_size=(100, 200)):
    """
    Extract individual pieces from a chess board image.
    
    Args:
        board_image_path: Path to the full board image
        output_dir: Directory to save extracted pieces
        piece_size: Size to resize pieces to (width, height)
    
    Returns:
        List of extracted piece paths
    """
    try:
        # Load the board image
        board_img = cv2.imread(board_image_path)
        if board_img is None:
            logger.error(f"Could not load image: {board_image_path}")
            return []
        
        # Convert BGR to RGB
        board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = board_img.shape[:2]
        
        # Calculate square size (assuming 8x8 grid)
        square_size = min(width, height) // 8
        
        extracted_pieces = []
        
        # Extract each square (64 squares total)
        for rank in range(8):
            for file in range(8):
                # Calculate square coordinates
                x1 = file * square_size
                y1 = rank * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Extract square
                square = board_img[y1:y2, x1:x2]
                
                if square.size > 0:
                    # Resize to standard piece size
                    square_resized = cv2.resize(square, piece_size)
                    
                    # Convert to PIL Image for saving
                    square_pil = Image.fromarray(square_resized)
                    
                    # Generate filename
                    base_name = Path(board_image_path).stem
                    piece_filename = f"{base_name}_{chr(97+file)}{8-rank}.png"
                    
                    # Save piece
                    piece_path = os.path.join(output_dir, piece_filename)
                    square_pil.save(piece_path, "PNG")
                    
                    extracted_pieces.append(piece_path)
        
        logger.info(f"Extracted {len(extracted_pieces)} pieces from {board_image_path}")
        return extracted_pieces
        
    except Exception as e:
        logger.error(f"Error processing {board_image_path}: {e}")
        return []

def organize_pieces_by_type(pieces_dir, output_base_dir):
    """
    Organize extracted pieces into the training dataset structure.
    This is a placeholder - you'll need to manually classify pieces or use existing models.
    
    Args:
        pieces_dir: Directory containing extracted pieces
        output_base_dir: Base directory for organized pieces
    """
    # Create output directories
    piece_types = [
        'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king',
        'white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king'
    ]
    
    for piece_type in piece_types:
        os.makedirs(os.path.join(output_base_dir, piece_type), exist_ok=True)
    
    # For now, we'll put all pieces in a temporary directory
    # You'll need to manually classify them or use your existing models
    temp_dir = os.path.join(output_base_dir, "temp_unclassified")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all pieces to temp directory
    piece_files = [f for f in os.listdir(pieces_dir) if f.endswith('.png')]
    for piece_file in piece_files:
        src = os.path.join(pieces_dir, piece_file)
        dst = os.path.join(temp_dir, piece_file)
        shutil.move(src, dst)
    
    logger.info(f"Moved {len(piece_files)} pieces to {temp_dir}")
    logger.info("You'll need to manually classify these pieces or use existing models to classify them.")

def main():
    """Main function to extract pieces from additional training images."""
    
    # Configuration
    input_dir = os.path.expanduser("~/Desktop/training_images_3")
    output_dir = "extracted_pieces_temp"
    final_output_dir = "grey_background_dataset/pieces/train"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of board images
    board_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"Found {len(board_images)} board images to process")
    
    # Process each board image
    total_pieces = 0
    for i, board_image in enumerate(board_images):
        logger.info(f"Processing {i+1}/{len(board_images)}: {board_image}")
        
        board_path = os.path.join(input_dir, board_image)
        pieces = extract_pieces_from_board(board_path, output_dir)
        total_pieces += len(pieces)
    
    logger.info(f"Extraction complete! Total pieces extracted: {total_pieces}")
    
    # Organize pieces
    logger.info("Organizing pieces...")
    organize_pieces_by_type(output_dir, final_output_dir)
    
    logger.info(f"✅ Process complete!")
    logger.info(f"📁 Extracted pieces are in: {output_dir}")
    logger.info(f"📁 Unclassified pieces are in: {final_output_dir}/temp_unclassified")
    logger.info(f"🔧 Next step: Classify the extracted pieces by piece type and color")

if __name__ == "__main__":
    main()

