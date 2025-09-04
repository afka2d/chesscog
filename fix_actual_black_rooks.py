#!/usr/bin/env python3
"""
Fix the actual black rook pieces in the TEST set.
This script will replace the pieces that actually contain black rooks according to the FEN.
"""

import os
import json
import cv2
import numpy as np
import chess
from pathlib import Path

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

def extract_specific_pieces(image_path, corners, fen, target_squares):
    """Extract only the specific pieces we need to replace."""
    print(f"🔧 Extracting black rook pieces from {os.path.basename(image_path)}...")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Define source and destination points for perspective transform
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
    
    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, (400, 400))
    
    # Parse FEN
    try:
        board = chess.Board(fen)
    except:
        print(f"❌ Invalid FEN: {fen}")
        return None
    
    # Create extraction directory
    extract_dir = f"temp_extract_{os.path.splitext(os.path.basename(image_path))[0]}"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract only the target pieces
    extracted_pieces = {}
    for square_name in target_squares:
        try:
            square = chess.parse_square(square_name)
            piece = board.piece_at(square)
            
            if piece is not None and piece.color == chess.BLACK and piece.symbol().lower() == 'r':
                # Calculate crop coordinates (50x50 pieces)
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)  # Convert to our coordinate system
                
                x1 = file * 50
                y1 = rank * 50
                x2 = x1 + 50
                y2 = y1 + 50
                
                # Crop piece
                piece_img = warped[y1:y2, x1:x2]
                
                # Save piece
                piece_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{square_name}.png"
                piece_path = os.path.join(extract_dir, piece_filename)
                cv2.imwrite(piece_path, piece_img)
                
                extracted_pieces[square_name] = {
                    'filename': piece_filename,
                    'piece_type': 'black_rook',
                    'path': piece_path
                }
                print(f"   ✅ Extracted {square_name} -> black_rook")
            else:
                print(f"   ⚠️  {square_name} is not a black rook")
        except Exception as e:
            print(f"   ❌ Error extracting {square_name}: {e}")
    
    print(f"   📊 Extracted {len(extracted_pieces)} black rook pieces to: {extract_dir}")
    return extract_dir, extracted_pieces

def replace_specific_pieces(extract_dir, extracted_pieces, image_name, dataset_type):
    """Replace only the specific pieces in the dataset."""
    print(f"🔧 Replacing black rook pieces for {image_name}...")
    
    replaced_count = 0
    backup_count = 0
    
    for square_name, piece_info in extracted_pieces.items():
        source_path = piece_info['path']
        target_path = f"grey_background_dataset/pieces/{dataset_type}/{piece_info['piece_type']}/{piece_info['filename']}"
        
        # Create backup if target exists
        if os.path.exists(target_path):
            backup_path = target_path + ".backup_rook_fix"
            os.rename(target_path, backup_path)
            backup_count += 1
        
        # Copy new piece
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.system(f"cp '{source_path}' '{target_path}'")
        replaced_count += 1
        print(f"   ✅ {piece_info['filename']} -> {piece_info['piece_type']}/")
    
    print(f"   📊 Replaced: {replaced_count} pieces")
    print(f"   💾 Backups created: {backup_count} files")
    return True

def fix_black_rooks(image_name, dataset_type):
    """Fix black rook pieces for an image."""
    print(f"\n🔧 FIXING BLACK ROOKS: {image_name} ({dataset_type.upper()} set)")
    print("=" * 60)
    
    # Load annotation
    annotation = load_annotation(image_name, dataset_type)
    if not annotation:
        print(f"❌ Could not load annotation for {image_name}")
        return False
    
    print(f"   📊 Current corners: {annotation['corners']}")
    print(f"   📝 Current FEN: {annotation['fen']}")
    
    # Find black rooks
    black_rooks = find_black_rooks(annotation['fen'])
    if not black_rooks:
        print(f"   ⚠️  No black rooks found in this position")
        return True  # Not an error, just no rooks to fix
    
    print(f"   ♜ Black rooks on: {', '.join(black_rooks)}")
    
    # Check if image exists
    image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Extract black rook pieces
    result = extract_specific_pieces(image_path, annotation['corners'], annotation['fen'], black_rooks)
    if not result:
        return False
    
    extract_dir, extracted_pieces = result
    
    # Replace black rook pieces
    success = replace_specific_pieces(extract_dir, extracted_pieces, image_name, dataset_type)
    
    # Cleanup
    print(f"🧹 Cleaning up temporary files...")
    os.system(f"rm -rf {extract_dir}")
    print(f"   🗑️  Removed: {extract_dir}/")
    
    if success:
        print(f"🎉 {image_name} black rooks fix complete!")
        return True
    else:
        print(f"❌ {image_name} black rooks fix failed!")
        return False

def main():
    """Fix the actual black rook pieces in the TEST set."""
    print("🔧 FIXING ACTUAL BLACK ROOK PIECES")
    print("=" * 50)
    print("This script will replace the pieces that actually contain black rooks")
    print("according to the corrected FENs.")
    print()
    
    # Images to check
    images_to_fix = [
        "NEW_20250805_135338_000",
        "NEW_20250805_135338_001", 
        "NEW_20250805_135338_002",
        "NEW_20250805_135338_003",
        "NEW_20250805_135338_004",
        "NEW_20250805_135338_005",
        "NEW_20250805_135338_006"
    ]
    
    success_count = 0
    total_rooks = 0
    
    for image_name in images_to_fix:
        # Count rooks first
        annotation = load_annotation(image_name, "test")
        if annotation:
            rooks = find_black_rooks(annotation['fen'])
            total_rooks += len(rooks)
        
        if fix_black_rooks(image_name, "test"):
            success_count += 1
    
    print(f"\n🎉 BLACK ROOK FIXING COMPLETE!")
    print(f"📊 Results:")
    print(f"   Total images processed: {len(images_to_fix)}")
    print(f"   Successful fixes: {success_count}")
    print(f"   Failed fixes: {len(images_to_fix) - success_count}")
    print(f"   Total black rooks replaced: {total_rooks}")

if __name__ == "__main__":
    main()
