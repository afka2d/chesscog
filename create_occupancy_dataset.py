#!/usr/bin/env python3
"""
Custom script to create occupancy dataset from grey_background_dataset
"""

import os
import json
import cv2
import numpy as np
import chess
from pathlib import Path
from PIL import Image
import shutil

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crop a chess square from the warped input image for occupancy classification."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]

def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image to a square chess board."""
    corners = np.array(corners, dtype=np.float32)
    destination = np.array([
        [SQUARE_SIZE, SQUARE_SIZE],
        [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],
        [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE],
        [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]
    ], dtype=np.float32)
    
    transformation_matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))

def create_occupancy_dataset():
    """Create occupancy dataset from grey_background_dataset."""
    input_base = Path("grey_background_dataset")
    output_base = Path("data:") / "occupancy"
    
    # Create output directories
    for subset in ("train", "val", "test"):
        for c in ("empty", "occupied"):
            folder = output_base / subset / c
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_squares = 0
    
    for subset in ("train", "val", "test"):
        images_dir = input_base / "images" / subset
        annotations_dir = input_base / "annotations" / subset
        
        if not images_dir.exists():
            print(f"Skipping {subset} - directory doesn't exist")
            continue
            
        image_files = list(images_dir.glob("*.JPG"))
        print(f"\nProcessing {len(image_files)} images in {subset} set...")
        
        for img_file in image_files:
            annotation_file = annotations_dir / (img_file.stem + ".json")
            
            if not annotation_file.exists():
                print(f"Warning: No annotation for {img_file.name}")
                continue
                
            try:
                # Load annotation
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                fen = annotation.get('fen', '')
                white_turn = annotation.get('white_turn', True)
                
                if not corners or len(corners) != 4:
                    print(f"Warning: Invalid corners for {img_file.name}")
                    continue
                    
                if not fen or fen == "8/8/8/8/8/8/8/8 w - - 0 1":
                    print(f"Warning: Empty FEN for {img_file.name}")
                    continue
                
                # Load and process image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Warning: Could not load {img_file.name}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Warp the chessboard
                unwarped = warp_chessboard_image(img, corners)
                
                # Parse FEN
                board = chess.Board(fen)
                
                # Extract squares
                for square in chess.SQUARES:
                    target_class = "empty" if board.piece_at(square) is None else "occupied"
                    square_img = crop_square(unwarped, square, white_turn)
                    
                    # Save square image
                    output_file = output_base / subset / target_class / f"{img_file.stem}_{chess.square_name(square)}.png"
                    with Image.fromarray(square_img, "RGB") as pil_img:
                        pil_img.save(output_file)
                    
                    total_squares += 1
                
                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Processed {total_processed}/{len(image_files)} images in {subset}")
                    
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
    
    # Print summary
    print(f"\n✅ Successfully created occupancy dataset!")
    print(f"Total images processed: {total_processed}")
    print(f"Total squares extracted: {total_squares}")
    
    for subset in ("train", "val", "test"):
        empty_count = len(list((output_base / subset / "empty").glob("*.png")))
        occupied_count = len(list((output_base / subset / "occupied").glob("*.png")))
        print(f"{subset}: {empty_count} empty, {occupied_count} occupied")

if __name__ == "__main__":
    create_occupancy_dataset()