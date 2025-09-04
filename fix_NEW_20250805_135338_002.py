#!/usr/bin/env python3
"""
Manually fix corners and FEN for NEW_20250805_135338_002.JPG and regenerate piece images.
"""

import cv2
import numpy as np
import json
import os
import shutil
from pathlib import Path

def display_image_with_corners(image_path, corners):
    """Display the image with current corners and allow manual adjustment."""
    print(f"🔍 Displaying image: {image_path}")
    print(f"📐 Current corners: {corners}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}")
        return None
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw current corners
    image_with_corners = image_rgb.copy()
    for i, corner in enumerate(corners):
        x, y = corner
        cv2.circle(image_with_corners, (x, y), 20, (255, 0, 0), -1)  # Blue circle
        cv2.putText(image_with_corners, str(i), (x+25, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Draw corner labels
    corner_labels = ['a8', 'h8', 'h1', 'a1']
    for i, (corner, label) in enumerate(zip(corners, corner_labels)):
        x, y = corner
        cv2.putText(image_with_corners, label, (x-30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display image
    cv2.imshow('NEW_20250805_135338_002 - Adjust Corners', cv2.cvtColor(image_with_corners, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image

def get_manual_corners():
    """Get manually adjusted corners from user."""
    print("\n📐 Manual Corner Adjustment")
    print("=" * 50)
    print("Please click on the four corners in this order:")
    print("1. a8 (top-left)")
    print("2. h8 (top-right)") 
    print("3. h1 (bottom-right)")
    print("4. a1 (bottom-left)")
    print("\nClick on the image to set each corner position.")
    
    # Read image for corner selection
    image_path = "grey_background_dataset/images/test/NEW_20250805_135338_002.JPG"
    image = cv2.imread(image_path)
    
    corners = []
    current_corner = 0
    corner_names = ['a8', 'h8', 'h1', 'a1']
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal corners, current_corner
        if event == cv2.EVENT_LBUTTONDOWN and current_corner < 4:
            corners.append([x, y])
            print(f"   Corner {current_corner + 1} ({corner_names[current_corner]}): [{x}, {y}]")
            current_corner += 1
            
            # Draw the corner
            cv2.circle(image, (x, y), 15, (0, 255, 0), -1)
            cv2.putText(image, corner_names[current_corner - 1], (x+20, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Select Corners', image)
            
            if current_corner == 4:
                print("   ✅ All corners selected!")
                cv2.waitKey(1000)  # Show final result briefly
    
    # Create window and set mouse callback
    cv2.namedWindow('Select Corners')
    cv2.setMouseCallback('Select Corners', mouse_callback)
    
    # Display image
    cv2.imshow('Select Corners', image)
    print("   Click on the image to set corners...")
    
    # Wait for all corners to be selected
    while current_corner < 4:
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    print(f"\n📐 Final corners: {corners}")
    return corners

def get_manual_fen():
    """Get manually input FEN from user."""
    print("\n📝 Manual FEN Input")
    print("=" * 30)
    
    # Show current FEN for reference
    current_fen = "3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1"
    print(f"Current FEN: {current_fen}")
    
    # Ask user to input correct FEN
    print("\nPlease enter the correct FEN for this position:")
    print("(or press Enter to keep current FEN)")
    
    user_fen = input("FEN: ").strip()
    
    if user_fen:
        print(f"✅ New FEN: {user_fen}")
        return user_fen
    else:
        print(f"✅ Keeping current FEN: {current_fen}")
        return current_fen

def extract_pieces_with_corners(image_path, corners, fen):
    """Extract individual piece images using the corrected corners."""
    print(f"\n🔄 Extracting pieces with corrected corners...")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}")
        return
    
    # Convert corners to numpy array
    corners_np = np.array(corners, dtype=np.float32)
    
    # Define target corners (perfect square)
    target_size = 400
    target_corners = np.array([
        [0, 0],                    # a8 (top-left)
        [target_size, 0],          # h8 (top-right)
        [target_size, target_size], # h1 (bottom-right)
        [0, target_size]           # a1 (bottom-left)
    ], dtype=np.float32)
    
    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(corners_np, target_corners)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(image, matrix, (target_size, target_size))
    
    # Save warped image for verification
    debug_path = "debug_outputs/NEW_20250805_135338_002_warped.png"
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite(debug_path, warped)
    print(f"   💾 Warped board saved to: {debug_path}")
    
    # Parse FEN to get piece positions
    import chess
    board = chess.Board(fen)
    
    # Piece type mapping
    piece_mapping = {
        'P': 'white_pawn', 'R': 'white_rook', 'N': 'white_knight',
        'B': 'white_bishop', 'Q': 'white_queen', 'K': 'white_king',
        'p': 'black_pawn', 'r': 'black_rook', 'n': 'black_knight',
        'b': 'black_bishop', 'q': 'black_queen', 'k': 'black_king'
    }
    
    # Create output directory
    output_dir = "re_extracted_NEW_20250805_135338_002"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract each square
    square_size = target_size // 8
    extracted_count = 0
    
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)  # Convert to chess coordinates
            piece = board.piece_at(square)
            
            if piece is not None:
                # Calculate square boundaries
                x1 = file * square_size
                y1 = rank * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Extract square
                square_img = warped[y1:y2, x1:x2]
                
                # Determine piece type and color
                piece_char = piece.symbol()
                folder_name = piece_mapping[piece_char]
                
                # Save piece image
                filename = f"NEW_20250805_135338_002_{chr(97+file)}{8-rank}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, square_img)
                
                print(f"   ✅ Extracted {piece_char} from {chr(97+file)}{8-rank} -> {filename}")
                extracted_count += 1
    
    print(f"\n🎯 Extracted {extracted_count} pieces to: {output_dir}")
    return output_dir

def update_annotation_file(corners, fen):
    """Update the annotation file with corrected corners and FEN."""
    print(f"\n📝 Updating annotation file...")
    
    annotation_path = "grey_background_dataset/annotations/test/NEW_20250805_135338_002.json"
    
    # Backup original annotation
    backup_path = "grey_background_dataset/annotations/test/NEW_20250805_135338_002.json.backup"
    if os.path.exists(annotation_path):
        shutil.copy2(annotation_path, backup_path)
        print(f"   💾 Original annotation backed up to: {backup_path}")
    
    # Create new annotation
    annotation = {
        "image": "NEW_20250805_135338_002.JPG",
        "corners": corners,
        "fen": fen,
        "white_turn": True,
        "timestamp": "manually_corrected"
    }
    
    # Save corrected annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"   ✅ Corrected annotation saved to: {annotation_path}")

def main():
    """Main function to fix NEW_20250805_135338_002."""
    print("🔧 Manual Fix for NEW_20250805_135338_002 - Corners and FEN")
    print("=" * 70)
    
    try:
        image_path = "grey_background_dataset/images/test/NEW_20250805_135338_002.JPG"
        
        # Step 1: Display current image with corners
        print("🔍 Step 1: Displaying current image with corners...")
        current_corners = [536, 1882], [2718, 1822], [2858, 4146], [356, 4088]
        image = display_image_with_corners(image_path, current_corners)
        
        # Step 2: Get manually adjusted corners
        print("\n🔍 Step 2: Manual corner adjustment...")
        corrected_corners = get_manual_corners()
        
        # Step 3: Get manually input FEN
        print("\n🔍 Step 3: Manual FEN input...")
        corrected_fen = get_manual_fen()
        
        # Step 4: Extract pieces with corrected corners
        print("\n🔍 Step 4: Extracting pieces with corrected corners...")
        output_dir = extract_pieces_with_corners(image_path, corrected_corners, corrected_fen)
        
        # Step 5: Update annotation file
        print("\n🔍 Step 5: Updating annotation file...")
        update_annotation_file(corrected_corners, corrected_fen)
        
        print(f"\n✅ Fix complete!")
        print(f"📐 Corrected corners: {corrected_corners}")
        print(f"📝 Corrected FEN: {corrected_fen}")
        print(f"🖼️  Pieces extracted to: {output_dir}")
        print(f"💾 Original annotation backed up")
        
        print(f"\n🔍 Next steps:")
        print(f"   1. Review the extracted pieces in: {output_dir}")
        print(f"   2. Verify the warped board: debug_outputs/NEW_20250805_135338_002_warped.png")
        print(f"   3. If satisfied, replace the original piece images in the dataset")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
