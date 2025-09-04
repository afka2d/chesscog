#!/usr/bin/env python3
"""
Find which squares actually contain black rooks in the problematic images.
This will help identify the correct squares to fix.
"""

import os
import json
import chess

def load_annotation(image_name, dataset_type):
    """Load annotation file for an image."""
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            return json.load(f)
    return None

def find_black_rooks(fen):
    """Find all squares that contain black rooks according to the FEN."""
    try:
        board = chess.Board(fen)
        black_rooks = []
        
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                
                if piece is not None and piece.color == chess.BLACK and piece.symbol().lower() == 'r':
                    square_name = chess.square_name(square)
                    black_rooks.append(square_name)
        
        return black_rooks
    except:
        return []

def main():
    """Find black rooks in the problematic images."""
    print("🔍 FINDING BLACK ROOKS IN PROBLEMATIC IMAGES")
    print("=" * 50)
    print("This will show which squares actually contain black rooks")
    print("according to the corrected FENs.")
    print()
    
    # Images to check
    images_to_check = [
        "NEW_20250805_135338_000",
        "NEW_20250805_135338_001", 
        "NEW_20250805_135338_002",
        "NEW_20250805_135338_003",
        "NEW_20250805_135338_004",
        "NEW_20250805_135338_005",
        "NEW_20250805_135338_006"
    ]
    
    for image_name in images_to_check:
        print(f"📊 {image_name}:")
        
        # Load annotation
        annotation = load_annotation(image_name, "test")
        if not annotation:
            print(f"   ❌ No annotation found")
            continue
        
        fen = annotation['fen']
        print(f"   📝 FEN: {fen}")
        
        # Find black rooks
        black_rooks = find_black_rooks(fen)
        if black_rooks:
            print(f"   ♜ Black rooks on: {', '.join(black_rooks)}")
        else:
            print(f"   ⚠️  No black rooks found in this position")
        
        print()

if __name__ == "__main__":
    main()
