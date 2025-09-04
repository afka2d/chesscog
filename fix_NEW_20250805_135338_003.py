#!/usr/bin/env python3
"""
Fix NEW_20250805_135338_003 by re-entering corners and FEN.
The FEN currently says there are black bishops on e6 and f6, but you see blank squares.
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

def get_manual_corners(image_path):
    """Interactive corner selection."""
    print(f"🔍 Loading image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Display image for corner selection
    cv2.namedWindow('Corner Selection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Corner Selection', 1200, 800)
    
    corners = []
    corner_names = ['a8 (top-left)', 'h8 (top-right)', 'h1 (bottom-right)', 'a1 (bottom-left)']
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x, y])
            print(f"   Corner {len(corners)} ({corner_names[len(corners)-1]}): [{x}, {y}]")
            
            # Draw the corner
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(img, f"{len(corners)}", (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Corner Selection', img)
    
    cv2.setMouseCallback('Corner Selection', mouse_callback)
    
    print(f"\n📋 CORNER SELECTION INSTRUCTIONS for {os.path.basename(image_path)}:")
    print(f"   Click on the four board corners in this order:")
    for i, name in enumerate(corner_names, 1):
        print(f"   {i}. {name}")
    print(f"   Press 'q' to quit, 'r' to reset")
    
    cv2.imshow('Corner Selection', img)
    
    while len(corners) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):
            corners = []
            img = cv2.imread(image_path)
            cv2.imshow('Corner Selection', img)
            print("   Reset - click corners again")
    
    cv2.destroyAllWindows()
    
    # Calculate board dimensions
    width = max(corners[1][0] - corners[0][0], corners[2][0] - corners[3][0])
    height = max(corners[3][1] - corners[0][1], corners[2][1] - corners[1][1])
    aspect_ratio = width / height if height > 0 else 1
    
    print(f"✅ Corner selection complete!")
    print(f"   Board dimensions: {width} x {height}")
    print(f"   Aspect ratio: {aspect_ratio:.3f}")
    
    if aspect_ratio < 0.8 or aspect_ratio > 1.25:
        print(f"   ⚠️  Warning: Board aspect ratio is {aspect_ratio:.3f}")
    else:
        print(f"   ✅ Good! Board is reasonably square")
    
    return corners

def get_manual_fen():
    """Interactive FEN input."""
    print(f"\n📝 FEN INPUT:")
    print(f"   Please enter the correct FEN for this position.")
    print(f"   Pay special attention to squares e6 and f6 - they should be empty (8) if you see blank squares.")
    print(f"   FEN format: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'")
    print(f"   (8 = empty squares, lowercase = black pieces, uppercase = white pieces)")
    print()
    
    while True:
        fen = input("   Enter FEN: ").strip()
        
        if not fen:
            print("   ❌ Please enter a FEN")
            continue
        
        try:
            # Validate FEN
            board = chess.Board(fen)
            print(f"   ✅ FEN accepted: {fen}")
            
            # Check e6 and f6 specifically
            e6_piece = board.piece_at(chess.parse_square('e6'))
            f6_piece = board.piece_at(chess.parse_square('f6'))
            
            e6_desc = 'Empty' if e6_piece is None else f"{'White' if e6_piece.color else 'Black'} {e6_piece.symbol().lower()}"
            f6_desc = 'Empty' if f6_piece is None else f"{'White' if f6_piece.color else 'Black'} {f6_piece.symbol().lower()}"
            print(f"   📊 e6: {e6_desc}")
            print(f"   📊 f6: {f6_desc}")
            
            return fen
        except:
            print(f"   ❌ Invalid FEN format. Please try again.")

def extract_pieces_with_corners(image_path, corners, fen):
    """Extract individual piece images using corrected corners and FEN."""
    print(f"🔧 Extracting pieces with corrected corners...")
    
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
    
    # Save warped board for reference
    warped_path = f"debug_outputs/{os.path.splitext(os.path.basename(image_path))[0]}_warped.png"
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite(warped_path, warped)
    print(f"   💾 Saved warped board: {warped_path}")
    
    # Parse FEN
    try:
        board = chess.Board(fen)
    except:
        print(f"❌ Invalid FEN: {fen}")
        return None
    
    # Create extraction directory
    extract_dir = f"re_extracted_{os.path.splitext(os.path.basename(image_path))[0]}"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract pieces
    piece_count = 0
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)  # Convert to chess square
            piece = board.piece_at(square)
            
            if piece is not None:
                # Calculate crop coordinates (50x50 pieces)
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
                square_name = chess.square_name(square)
                piece_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{square_name}.png"
                piece_path = os.path.join(extract_dir, piece_filename)
                cv2.imwrite(piece_path, piece_img)
                piece_count += 1
    
    print(f"   ✅ Extracted {piece_count} piece images to: {extract_dir}")
    return extract_dir

def update_annotation_file(corners, fen, image_name, dataset_type):
    """Update the annotation file with new corners and FEN."""
    print(f"📝 Updating annotation file...")
    
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    
    # Create backup
    backup_path = annotation_path + ".backup_fen_fix"
    if os.path.exists(annotation_path):
        os.rename(annotation_path, backup_path)
        print(f"   💾 Created backup: {backup_path}")
    
    # Create new annotation
    annotation = {
        "image": f"{image_name}.JPG",
        "corners": corners,
        "fen": fen,
        "white_turn": True,  # Default, can be adjusted if needed
        "timestamp": "fen_corrected"
    }
    
    # Save new annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"   ✅ Updated annotation: {annotation_path}")
    print(f"   📊 New corners: {corners}")
    print(f"   📝 New FEN: {fen}")

def main():
    """Fix NEW_20250805_135338_003 by re-entering corners and FEN."""
    print("🔧 FIXING NEW_20250805_135338_003")
    print("=" * 50)
    print("The FEN currently says there are black bishops on e6 and f6,")
    print("but you see blank squares. Let's correct this.")
    print()
    
    image_name = "NEW_20250805_135338_003"
    dataset_type = "test"
    image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load current annotation
    current_annotation = load_annotation(image_name, dataset_type)
    if current_annotation:
        print(f"📊 Current corners: {current_annotation['corners']}")
        print(f"📝 Current FEN: {current_annotation['fen']}")
    else:
        print(f"❌ No current annotation found")
    
    print(f"\n🎯 STEP 1: Manual Corner Selection")
    corners = get_manual_corners(image_path)
    if corners is None:
        print(f"❌ Corner selection cancelled")
        return
    
    print(f"\n🎯 STEP 2: FEN Input")
    fen = get_manual_fen()
    
    print(f"\n🎯 STEP 3: Piece Extraction")
    extract_dir = extract_pieces_with_corners(image_path, corners, fen)
    if extract_dir is None:
        print(f"❌ Piece extraction failed")
        return
    
    print(f"\n🎯 STEP 4: Update Annotation")
    update_annotation_file(corners, fen, image_name, dataset_type)
    
    print(f"\n🎉 NEW_20250805_135338_003 fix complete!")
    print(f"   📊 New corners: {corners}")
    print(f"   📝 New FEN: {fen}")
    print(f"   📁 Extracted pieces: {extract_dir}")
    print(f"   💾 Warped board saved for verification")

if __name__ == "__main__":
    main()
