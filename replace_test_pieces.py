#!/usr/bin/env python3
"""
Replace specific TEST set pieces that still show issues.
This script will re-extract and replace only the problematic pieces mentioned by the user.
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

def extract_specific_pieces(image_path, corners, fen, target_squares):
    """Extract only the specific pieces we need to replace."""
    print(f"🔧 Extracting specific pieces from {os.path.basename(image_path)}...")
    
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
            
            if piece is not None:
                # Calculate crop coordinates (50x50 pieces)
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)  # Convert to our coordinate system
                
                x1 = file * 50
                y1 = rank * 50
                x2 = x1 + 50
                y2 = y1 + 50
                
                # Crop piece
                piece_img = warped[y1:y2, x1:x2]
                
                # Determine piece type
                piece_type = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                if piece.symbol().lower() == 'p':
                    piece_type = f"{'white' if piece.color else 'black'}_pawn"
                elif piece.symbol().lower() == 'r':
                    piece_type = f"{'white' if piece.color else 'black'}_rook"
                elif piece.symbol().lower() == 'n':
                    piece_type = f"{'white' if piece.color else 'black'}_knight"
                elif piece.symbol().lower() == 'b':
                    piece_type = f"{'white' if piece.color else 'black'}_bishop"
                elif piece.symbol().lower() == 'q':
                    piece_type = f"{'white' if piece.color else 'black'}_queen"
                elif piece.symbol().lower() == 'k':
                    piece_type = f"{'white' if piece.color else 'black'}_king"
                
                # Save piece
                piece_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{square_name}.png"
                piece_path = os.path.join(extract_dir, piece_filename)
                cv2.imwrite(piece_path, piece_img)
                
                extracted_pieces[square_name] = {
                    'filename': piece_filename,
                    'piece_type': piece_type,
                    'path': piece_path
                }
                print(f"   ✅ Extracted {square_name} -> {piece_type}")
            else:
                print(f"   ⚠️  No piece on {square_name}")
        except Exception as e:
            print(f"   ❌ Error extracting {square_name}: {e}")
    
    print(f"   📊 Extracted {len(extracted_pieces)} pieces to: {extract_dir}")
    return extract_dir, extracted_pieces

def replace_specific_pieces(extract_dir, extracted_pieces, image_name, dataset_type):
    """Replace only the specific pieces in the dataset."""
    print(f"🔧 Replacing specific pieces for {image_name}...")
    
    replaced_count = 0
    backup_count = 0
    
    for square_name, piece_info in extracted_pieces.items():
        source_path = piece_info['path']
        target_path = f"grey_background_dataset/pieces/{dataset_type}/{piece_info['piece_type']}/{piece_info['filename']}"
        
        # Create backup if target exists
        if os.path.exists(target_path):
            backup_path = target_path + ".backup_final_fix"
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

def fix_specific_pieces(image_name, dataset_type, target_squares):
    """Fix specific pieces for an image."""
    print(f"\n🔧 FIXING SPECIFIC PIECES: {image_name} ({dataset_type.upper()} set)")
    print("=" * 60)
    
    # Load annotation
    annotation = load_annotation(image_name, dataset_type)
    if not annotation:
        print(f"❌ Could not load annotation for {image_name}")
        return False
    
    print(f"   📊 Current corners: {annotation['corners']}")
    print(f"   📝 Current FEN: {annotation['fen']}")
    print(f"   🎯 Target squares: {target_squares}")
    
    # Check if image exists
    image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Extract specific pieces
    result = extract_specific_pieces(image_path, annotation['corners'], annotation['fen'], target_squares)
    if not result:
        return False
    
    extract_dir, extracted_pieces = result
    
    # Replace specific pieces
    success = replace_specific_pieces(extract_dir, extracted_pieces, image_name, dataset_type)
    
    # Cleanup
    print(f"🧹 Cleaning up temporary files...")
    os.system(f"rm -rf {extract_dir}")
    print(f"   🗑️  Removed: {extract_dir}/")
    
    if success:
        print(f"🎉 {image_name} specific pieces fix complete!")
        return True
    else:
        print(f"❌ {image_name} specific pieces fix failed!")
        return False

def main():
    """Fix the specific TEST set pieces mentioned by the user."""
    print("🔧 REPLACING SPECIFIC TEST SET PIECES")
    print("=" * 50)
    print("This script will replace only the specific problematic pieces")
    print("mentioned by the user in the TEST set.")
    print()
    
    # Define the specific pieces to fix based on user feedback
    pieces_to_fix = {
        "NEW_20250805_135338_000": ["a8", "h5"],
        "NEW_20250805_135338_001": ["e8"],
        "NEW_20250805_135338_002": ["d8", "f8"],
        "NEW_20250805_135338_003": ["f8"],
        "NEW_20250805_135338_004": ["h8"],
        "NEW_20250805_135338_005": ["a8", "e7"],
        "NEW_20250805_135338_006": ["a8", "h5"]
    }
    
    success_count = 0
    total_pieces = sum(len(squares) for squares in pieces_to_fix.values())
    
    for image_name, target_squares in pieces_to_fix.items():
        if fix_specific_pieces(image_name, "test", target_squares):
            success_count += 1
    
    print(f"\n🎉 SPECIFIC PIECES REPLACEMENT COMPLETE!")
    print(f"📊 Results:")
    print(f"   Total images processed: {len(pieces_to_fix)}")
    print(f"   Successful fixes: {success_count}")
    print(f"   Failed fixes: {len(pieces_to_fix) - success_count}")
    print(f"   Total pieces replaced: {total_pieces}")

if __name__ == "__main__":
    main()
