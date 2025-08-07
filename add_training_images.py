#!/usr/bin/env python3
"""
Script to add chess board images to the grey background training dataset.
This script creates annotations for the images based on the provided descriptions.
"""

import os
import json
import shutil
from pathlib import Path

# Define the dataset directory
DATASET_DIR = "grey_background_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

# Image descriptions and their corresponding FEN notations
IMAGE_DATA = [
    # Original 10 images
    {
        "filename": "chess_starting_position_1.jpg",
        "description": "Standard starting position with all 32 pieces",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pieces": [
            # White pieces
            ("R", "a1"), ("N", "b1"), ("B", "c1"), ("Q", "d1"), ("K", "e1"), ("B", "f1"), ("N", "g1"), ("R", "h1"),
            ("P", "a2"), ("P", "b2"), ("P", "c2"), ("P", "d2"), ("P", "e2"), ("P", "f2"), ("P", "g2"), ("P", "h2"),
            # Black pieces
            ("r", "a8"), ("n", "b8"), ("b", "c8"), ("q", "d8"), ("k", "e8"), ("b", "f8"), ("n", "g8"), ("r", "h8"),
            ("p", "a7"), ("p", "b7"), ("p", "c7"), ("p", "d7"), ("p", "e7"), ("p", "f7"), ("p", "g7"), ("p", "h7")
        ]
    },
    {
        "filename": "chess_two_pawns_1.jpg",
        "description": "Two pawns: white pawn on e5, black pawn on d4",
        "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
        "pieces": [
            ("P", "e5"),  # White pawn
            ("p", "d4")   # Black pawn
        ]
    },
    
    # Additional images from descriptions
    {
        "filename": "chess_two_pawns_buschess.jpg",
        "description": "Two pawns with BUSCHESS branding: black pawn on d5, white pawn on e4",
        "fen": "8/8/8/3p4/4P3/8/8/8 w - - 0 1",
        "pieces": [
            ("p", "d5"),  # Black pawn
            ("P", "e4")   # White pawn
        ]
    },
    {
        "filename": "chess_two_pawns_rotated.jpg",
        "description": "Two pawns rotated board: black pawn on d5, white pawn on d4",
        "fen": "8/8/8/3p4/3P4/8/8/8 w - - 0 1",
        "pieces": [
            ("p", "d5"),  # Black pawn
            ("P", "d4")   # White pawn
        ]
    },
    {
        "filename": "chess_two_pawns_grey_bg.jpg",
        "description": "Two pawns on grey background: black pawn on e4, white pawn on f5",
        "fen": "8/8/8/8/4p3/5P2/8/8 w - - 0 1",
        "pieces": [
            ("p", "e4"),  # Black pawn
            ("P", "f5")   # White pawn
        ]
    },
    {
        "filename": "chess_complex_position_1.jpg",
        "description": "Complex position: white king on f1, 7 white pawns, black rook on a8, 4 black pawns",
        "fen": "r7/8/8/8/5p2/3p4/2pP4/5K2 w - - 0 1",
        "pieces": [
            # White pieces
            ("K", "f1"),  # White king
            ("P", "c4"), ("P", "d4"), ("P", "e4"), ("P", "d6"), ("P", "e5"), ("P", "g6"), ("P", "g7"),  # White pawns
            # Black pieces
            ("r", "a8"),  # Black rook
            ("p", "c3"), ("p", "d3"), ("p", "e3"), ("p", "f4")  # Black pawns
        ]
    },
    {
        "filename": "chess_five_pieces.jpg",
        "description": "Five pieces: white king on e2, white pawn on g6, black rook on b6, black rook on e4, black rook on b3",
        "fen": "8/8/1r6/8/4r3/1r6/4K3/8 w - - 0 1",
        "pieces": [
            ("K", "e2"),  # White king
            ("P", "g6"),  # White pawn
            ("r", "b6"),  # Black rook
            ("r", "e4"),  # Black rook
            ("r", "b3")   # Black rook
        ]
    },
    {
        "filename": "chess_five_pieces_2.jpg",
        "description": "Five pieces: white king on d2, white pawn on f6, black pawn on b6, black pawn on d4, black pawn on a3",
        "fen": "8/8/1p6/8/3p4/1p6/3K4/8 w - - 0 1",
        "pieces": [
            ("K", "d2"),  # White king
            ("P", "f6"),  # White pawn
            ("p", "b6"),  # Black pawn
            ("p", "d4"),  # Black pawn
            ("p", "a3")   # Black pawn
        ]
    },
    {
        "filename": "chess_nine_pieces.jpg",
        "description": "Nine pieces: white king on c5, white bishop on e4, 3 white pawns, black rook on a7, 3 black pawns",
        "fen": "r7/8/1p6/2K5/3Bp3/2P1P3/1P6/8 w - - 0 1",
        "pieces": [
            # White pieces
            ("K", "c5"),  # White king
            ("B", "e4"),  # White bishop
            ("P", "d3"), ("P", "f3"), ("P", "g2"),  # White pawns
            # Black pieces
            ("r", "a7"),  # Black rook
            ("p", "b6"), ("p", "d4"), ("p", "e6")  # Black pawns
        ]
    },
    {
        "filename": "chess_checkers_pieces.jpg",
        "description": "Nine checkers pieces: 5 white, 4 black on various squares",
        "fen": "8/7P/6P1/5P2/4P3/3P4/2P5/1P6 w - - 0 1",  # Simplified FEN for checkers
        "pieces": [
            # White checkers pieces
            ("P", "f1"), ("P", "f4"), ("P", "e5"), ("P", "d6"), ("P", "h7"),
            # Black checkers pieces
            ("p", "g1"), ("p", "e3"), ("p", "c3"), ("p", "c5")
        ]
    },
    {
        "filename": "chess_checkers_pieces_2.jpg",
        "description": "Nine checkers pieces: 5 white, 4 black on various squares (second angle)",
        "fen": "8/7P/6P1/5P2/4P3/3P4/2P5/1P6 w - - 0 1",  # Simplified FEN for checkers
        "pieces": [
            # White checkers pieces
            ("P", "f1"), ("P", "f4"), ("P", "e5"), ("P", "d6"), ("P", "h7"),
            # Black checkers pieces
            ("p", "g1"), ("p", "e3"), ("p", "c3"), ("p", "c5")
        ]
    },
    {
        "filename": "chess_five_pieces_3.jpg",
        "description": "Five pieces: white pawn on e6, white king on d3, black pawn on b7, black pawn on d5, black rook on g2 (lying down)",
        "fen": "8/1p6/4P3/3p4/8/3K4/6r1/8 w - - 0 1",
        "pieces": [
            ("P", "e6"),  # White pawn
            ("K", "d3"),  # White king
            ("p", "b7"),  # Black pawn
            ("p", "d5"),  # Black pawn
            ("r", "g2")   # Black rook (lying down)
        ]
    },
    {
        "filename": "chess_five_pieces_4.jpg",
        "description": "Five pieces: white pawn on g6, white king on e2, black pawn on c7, black pawn on e4, black rook on g2 (lying down)",
        "fen": "8/2p5/6P1/8/4p3/8/4K2r/8 w - - 0 1",
        "pieces": [
            ("P", "g6"),  # White pawn
            ("K", "e2"),  # White king
            ("p", "c7"),  # Black pawn
            ("p", "e4"),  # Black pawn
            ("r", "g2")   # Black rook (lying down)
        ]
    },
    {
        "filename": "chess_five_pieces_5.jpg",
        "description": "Five pieces: black pawn on b6, white pawn on f6, black rook on e4, black bishop on b3, white king on e2",
        "fen": "8/8/1p6/8/4r3/1b6/4K3/8 w - - 0 1",
        "pieces": [
            ("p", "b6"),  # Black pawn
            ("P", "f6"),  # White pawn
            ("r", "e4"),  # Black rook
            ("b", "b3"),  # Black bishop
            ("K", "e2")   # White king
        ]
    },
    {
        "filename": "chess_checkers_pieces_3.jpg",
        "description": "Nine checkers pieces: 5 white, 4 black on various squares (third angle)",
        "fen": "8/7P/6P1/5P2/4P3/3P4/2P5/1P6 w - - 0 1",  # Simplified FEN for checkers
        "pieces": [
            # White checkers pieces
            ("P", "f1"), ("P", "f4"), ("P", "e5"), ("P", "d6"), ("P", "h7"),
            # Black checkers pieces
            ("p", "g1"), ("p", "e3"), ("p", "c3"), ("p", "c5")
        ]
    },
    {
        "filename": "chess_nine_pawns.jpg",
        "description": "Nine pawns: 5 white pawns, 4 black pawns on rotated board",
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",  # Simplified FEN for pawns only
        "pieces": [
            # White pawns
            ("P", "c6"), ("P", "d5"), ("P", "e2"), ("P", "e6"), ("P", "g7"),
            # Black pawns
            ("p", "c4"), ("p", "c7"), ("p", "e4"), ("p", "g1")
        ]
    },
    {
        "filename": "chess_five_pieces_6.jpg",
        "description": "Five pieces: white pawn on e6, white king on d3, black pawn on b7, black pawn on d5, black rook on g2 (lying down)",
        "fen": "8/1p6/4P3/3p4/8/3K4/6r1/8 w - - 0 1",
        "pieces": [
            ("P", "e6"),  # White pawn
            ("K", "d3"),  # White king
            ("p", "b7"),  # Black pawn
            ("p", "d5"),  # Black pawn
            ("r", "g2")   # Black rook (lying down)
        ]
    },
    {
        "filename": "chess_checkers_pieces_4.jpg",
        "description": "Nine checkers pieces: 5 white, 4 black on various squares (fourth angle)",
        "fen": "8/7P/6P1/5P2/4P3/3P4/2P5/1P6 w - - 0 1",  # Simplified FEN for checkers
        "pieces": [
            # White checkers pieces
            ("P", "f1"), ("P", "f4"), ("P", "e5"), ("P", "d6"), ("P", "h7"),
            # Black checkers pieces
            ("p", "g1"), ("p", "e3"), ("p", "c3"), ("p", "c5")
        ]
    },
    {
        "filename": "chess_five_pieces_7.jpg",
        "description": "Five pieces: white pawn on e6, white king on d3, black pawn on b7, black pawn on d5, black rook on g2 (lying down)",
        "fen": "8/1p6/4P3/3p4/8/3K4/6r1/8 w - - 0 1",
        "pieces": [
            ("P", "e6"),  # White pawn
            ("K", "d3"),  # White king
            ("p", "b7"),  # Black pawn
            ("p", "d5"),  # Black pawn
            ("r", "g2")   # Black rook (lying down)
        ]
    },
    {
        "filename": "chess_checkers_pieces_5.jpg",
        "description": "Nine checkers pieces: 5 white, 4 black on various squares (fifth angle)",
        "fen": "8/7P/6P1/5P2/4P3/3P4/2P5/1P6 w - - 0 1",  # Simplified FEN for checkers
        "pieces": [
            # White checkers pieces
            ("P", "f1"), ("P", "f4"), ("P", "e5"), ("P", "d6"), ("P", "h7"),
            # Black checkers pieces
            ("p", "g1"), ("p", "e3"), ("p", "c3"), ("p", "c5")
        ]
    }
]

def create_annotation(image_data):
    """Create annotation JSON for an image."""
    annotation = {
        "image_name": image_data["filename"],
        "description": image_data["description"],
        "fen": image_data["fen"],
        "corners": {
            "top_left": [100, 100],
            "top_right": [700, 100],
            "bottom_right": [700, 700],
            "bottom_left": [100, 700]
        },
        "pieces": []
    }
    
    # Add pieces to annotation
    for piece_symbol, square in image_data["pieces"]:
        annotation["pieces"].append({
            "piece": piece_symbol,
            "square": square
        })
    
    return annotation

def main():
    """Main function to add images to the dataset."""
    print("Adding chess board images to grey background dataset...")
    
    # Ensure directories exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    # Create annotations for each image
    for image_data in IMAGE_DATA:
        filename = image_data["filename"]
        annotation = create_annotation(image_data)
        
        # Save annotation
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{Path(filename).stem}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"✓ Created annotation for {filename}")
        print(f"  FEN: {image_data['fen']}")
        print(f"  Pieces: {len(image_data['pieces'])} pieces")
        print()
    
    print(f"✓ Created {len(IMAGE_DATA)} annotations in {ANNOTATIONS_DIR}")
    print()
    print("Next steps:")
    print("1. Add your actual chess board images to the 'grey_background_dataset/images/' directory")
    print("2. Update the corner coordinates in the annotation files using the update_corners.py script")
    print("3. Verify the FEN notations are correct for your specific images")
    print("4. Run the training pipeline")
    print()
    print("To update corner coordinates interactively:")
    print("  python update_corners.py --interactive")
    print()
    print("To list all annotated images:")
    print("  python update_corners.py --list")

if __name__ == "__main__":
    main() 