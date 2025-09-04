#!/usr/bin/env python3
"""
Fix IMG_4752.JPG by correcting corner coordinates and regenerating piece images.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

def get_manual_corners(image_path):
    """Allow user to manually click on the four board corners."""
    print(f"🔍 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Resize for display if too large
    height, width = image.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_image = cv2.resize(image, (new_width, new_height))
        scale_factor = scale
    else:
        display_image = image.copy()
        scale_factor = 1.0
    
    corners = []
    corner_names = ['a8 (top-left)', 'h8 (top-right)', 'h1 (bottom-right)', 'a1 (bottom-left)']
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates back to original image coordinates
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            corners.append([orig_x, orig_y])
            
            # Draw the point on the display image
            cv2.circle(display_image, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(display_image, f"{len(corners)}", (x+15, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Click Board Corners', display_image)
            
            print(f"   Corner {len(corners)} ({corner_names[len(corners)-1]}): [{orig_x}, {orig_y}]")
    
    # Create window and set mouse callback
    cv2.namedWindow('Click Board Corners', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Click Board Corners', mouse_callback)
    
    # Display instructions
    print(f"\n📋 CORNER SELECTION INSTRUCTIONS:")
    print(f"   Click on the four board corners in this order:")
    print(f"   1. a8 (top-left corner)")
    print(f"   2. h8 (top-right corner)") 
    print(f"   3. h1 (bottom-right corner)")
    print(f"   4. a1 (bottom-left corner)")
    print(f"   Press 'q' to quit, 'r' to reset")
    
    cv2.imshow('Click Board Corners', display_image)
    
    while len(corners) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User cancelled corner selection")
        elif key == ord('r'):
            corners.clear()
            display_image = cv2.resize(image, (new_width, new_height)) if scale_factor != 1.0 else image.copy()
            cv2.imshow('Click Board Corners', display_image)
            print(f"   🔄 Reset - click corners again")
    
    cv2.destroyAllWindows()
    
    # Verify corners form a reasonable rectangle
    corners_np = np.array(corners, dtype=np.float32)
    board_width = max(corners_np[:, 0]) - min(corners_np[:, 0])
    board_height = max(corners_np[:, 1]) - min(corners_np[:, 1])
    aspect_ratio = board_width / board_height
    
    print(f"\n✅ Corner selection complete!")
    print(f"   Board dimensions: {board_width:.0f} x {board_height:.0f}")
    print(f"   Aspect ratio: {aspect_ratio:.3f}")
    
    if 0.95 <= aspect_ratio <= 1.05:
        print(f"   🎯 Excellent! Board is nearly square")
    elif 0.9 <= aspect_ratio <= 1.1:
        print(f"   ✅ Good! Board is reasonably square")
    else:
        print(f"   ⚠️  Warning: Board aspect ratio is {aspect_ratio:.3f}")
    
    return corners

def get_manual_fen():
    """Allow user to manually input the FEN."""
    print(f"\n📝 FEN INPUT:")
    print(f"   Current FEN: 8/3k4/2n1q3/1n1p1p2/4P3/2N2P2/PPP5/1N1Q4 w - - 0 1")
    print(f"   Please verify this FEN matches the actual board position")
    
    while True:
        try:
            fen = input("   Enter correct FEN (or press Enter to keep current): ").strip()
            
            if not fen:  # User wants to keep current FEN
                fen = "8/3k4/2n1q3/1n1p1p2/4P3/2N2P2/PPP5/1N1Q4 w - - 0 1"
                print(f"   ✅ Keeping current FEN: {fen}")
                break
            
            # Basic FEN validation
            if len(fen.split('/')) == 8:
                print(f"   ✅ FEN accepted: {fen}")
                break
            else:
                print(f"   ❌ Invalid FEN format. Please try again.")
                
        except KeyboardInterrupt:
            raise KeyboardInterrupt("User cancelled FEN input")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return fen

def extract_pieces_with_corners(image_path, corners, fen):
    """Extract individual piece images using the provided corners and FEN."""
    print(f"\n🔧 Extracting pieces with corrected corners...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Define target board size for warping
    target_size = 400
    
    # Define target corners (square board)
    target_corners = np.array([
        [0, 0],                    # a8 (top-left)
        [target_size, 0],          # h8 (top-right) 
        [target_size, target_size], # h1 (bottom-right)
        [0, target_size]           # a1 (bottom-left)
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    corners_np = np.array(corners, dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(corners_np, target_corners)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(image, transform_matrix, (target_size, target_size))
    
    # Save warped board for verification
    debug_dir = "debug_outputs"
    os.makedirs(debug_dir, exist_ok=True)
    warped_path = os.path.join(debug_dir, "IMG_4752_warped.png")
    cv2.imwrite(warped_path, warped)
    print(f"   💾 Saved warped board: {warped_path}")
    
    # Parse FEN to get piece positions
    piece_positions = {}
    fen_parts = fen.split()
    board_fen = fen_parts[0]
    
    rank = 8
    file = 0
    
    for char in board_fen:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            square = chr(ord('a') + file) + str(rank)
            piece_type = get_piece_type(char)
            piece_positions[square] = piece_type
            file += 1
    
    # Extract individual piece images
    square_size = target_size // 8
    output_dir = "re_extracted_IMG_4752"
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    for square, piece_type in piece_positions.items():
        if piece_type:  # Skip empty squares
            file_idx = ord(square[0]) - ord('a')
            rank_idx = 8 - int(square[1])
            
            # Calculate square boundaries
            x1 = file_idx * square_size
            y1 = rank_idx * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            # Extract square
            square_img = warped[y1:y2, x1:x2]
            
            # Save piece image
            filename = f"IMG_4752_{square}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, square_img)
            extracted_count += 1
    
    print(f"   ✅ Extracted {extracted_count} piece images to: {output_dir}")
    return output_dir

def get_piece_type(char):
    """Convert FEN character to piece type."""
    piece_map = {
        'K': 'white_king', 'Q': 'white_queen', 'R': 'white_rook',
        'B': 'white_bishop', 'N': 'white_knight', 'P': 'white_pawn',
        'k': 'black_king', 'q': 'black_queen', 'r': 'black_rook',
        'b': 'black_bishop', 'n': 'black_knight', 'p': 'black_pawn'
    }
    return piece_map.get(char, None)

def update_annotation_file(corners, fen):
    """Update the annotation file with corrected corners and FEN."""
    print(f"\n📝 Updating annotation file...")
    
    annotation_path = "grey_background_dataset/annotations/test/IMG_4752.json"
    
    # Create backup
    backup_path = annotation_path + ".backup_before_fix"
    if os.path.exists(annotation_path):
        import shutil
        shutil.copy2(annotation_path, backup_path)
        print(f"   💾 Created backup: {backup_path}")
    
    # Update annotation
    annotation = {
        "image": "IMG_4752.JPG",
        "corners": corners,
        "fen": fen,
        "white_turn": True,
        "timestamp": "corrected_corners_and_fen"
    }
    
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"   ✅ Updated annotation: {annotation_path}")
    print(f"   📊 New corners: {corners}")
    print(f"   📝 New FEN: {fen}")

def main():
    """Main function to fix IMG_4752.JPG."""
    print("🔧 IMG_4752.JPG Fixing Script")
    print("=" * 50)
    
    try:
        # Check if image exists
        image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"📸 Found image: {image_path}")
        
        # Step 1: Get manual corner coordinates
        print(f"\n🎯 STEP 1: Manual Corner Selection")
        corners = get_manual_corners(image_path)
        
        # Step 2: Get manual FEN
        print(f"\n🎯 STEP 2: FEN Verification")
        fen = get_manual_fen()
        
        # Step 3: Extract pieces with corrected corners
        print(f"\n🎯 STEP 3: Piece Extraction")
        output_dir = extract_pieces_with_corners(image_path, corners, fen)
        
        # Step 4: Update annotation file
        print(f"\n🎯 STEP 4: Update Annotation")
        update_annotation_file(corners, fen)
        
        print(f"\n🎉 IMG_4752.JPG fix complete!")
        print(f"   📁 Re-extracted pieces: {output_dir}")
        print(f"   📊 Corrected corners: {corners}")
        print(f"   📝 Verified FEN: {fen}")
        print(f"\n💡 Next steps:")
        print(f"   1. Verify the warped board looks correct")
        print(f"   2. Check the extracted piece images")
        print(f"   3. Replace dataset pieces if satisfied")
        
    except KeyboardInterrupt:
        print(f"\n❌ User cancelled the operation")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
