#!/usr/bin/env python3
"""
Re-extract all pieces from IMG_4762 using the corrected FEN and corners.
Show each piece image for verification.
"""

import os
import cv2
import numpy as np
import json
import chess
import matplotlib.pyplot as plt

def warp_chessboard(img, corners):
    """Warp the chessboard to a square grid using the provided corners."""
    # Convert corners to numpy array
    src_corners = np.array(corners, dtype=np.float32)
    
    # Define destination corners (square grid)
    board_size = 400  # Size of warped board
    dst_corners = np.array([
        [0, 0],                    # Top-left
        [board_size, 0],           # Top-right
        [board_size, board_size],  # Bottom-right
        [0, board_size]            # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform
    transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Apply transform
    warped = cv2.warpPerspective(img, transform_matrix, (board_size, board_size))
    
    return warped

def extract_and_show_pieces(img, corners, fen, image_name):
    """Extract individual pieces and show each one for verification."""
    try:
        # Parse FEN to get piece positions
        board = chess.Board(fen)
        
        # Warp the chessboard to get a square grid
        warped = warp_chessboard(img, corners)
        
        # Calculate square size
        square_size = min(warped.shape[0], warped.shape[1]) // 8
        
        pieces_extracted = 0
        
        # Piece type mapping
        piece_mapping = {
            'P': 'white_pawn', 'R': 'white_rook', 'N': 'white_knight',
            'B': 'white_bishop', 'Q': 'white_queen', 'K': 'white_king',
            'p': 'black_pawn', 'r': 'black_rook', 'n': 'black_knight',
            'b': 'black_bishop', 'q': 'black_queen', 'k': 'black_king'
        }
        
        # Create output directory for extracted pieces
        output_dir = "re_extracted_IMG_4762"
        os.makedirs(output_dir, exist_ok=True)
        
        # Store pieces for display
        piece_images = []
        piece_info = []
        
        # Process each square
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Convert to chess coordinates
                piece = board.piece_at(square)
                
                # Extract square image regardless of whether there's a piece
                x1 = file * square_size
                y1 = rank * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                square_img = warped[y1:y2, x1:x2]
                
                if square_img.size > 0:
                    # Resize to standard size
                    square_resized = cv2.resize(square_img, (100, 200))
                    
                    # Generate filename
                    piece_filename = f"{image_name}_{chr(97+file)}{8-rank}.png"
                    piece_path = os.path.join(output_dir, piece_filename)
                    
                    # Save square image
                    cv2.imwrite(piece_path, square_resized)
                    
                    if piece is not None:
                        # Determine piece type and color
                        piece_char = piece.symbol()
                        folder_name = piece_mapping[piece_char]
                        
                        # Store for display
                        piece_images.append(square_resized)
                        piece_info.append(f"{chr(97+file)}{8-rank}: {piece_char} -> {folder_name}")
                        
                        print(f"   Extracted {piece_char} from {chr(97+file)}{8-rank} -> {folder_name}/{piece_filename}")
                        pieces_extracted += 1
                    else:
                        # Empty square
                        piece_images.append(square_resized)
                        piece_info.append(f"{chr(97+file)}{8-rank}: EMPTY")
                        
                        print(f"   Extracted EMPTY from {chr(97+file)}{8-rank} -> {piece_filename}")
        
        # Show all extracted pieces in a grid
        show_extracted_pieces(piece_images, piece_info, output_dir)
        
        return pieces_extracted
        
    except Exception as e:
        print(f"❌ Error extracting pieces: {e}")
        return 0

def show_extracted_pieces(piece_images, piece_info, output_dir):
    """Show all extracted pieces in a grid layout."""
    if not piece_images:
        print("❌ No pieces to display")
        return
    
    # Calculate grid dimensions
    n_pieces = len(piece_images)
    cols = 8  # Always 8 columns for 8x8 board
    rows = 8
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    # Display each square
    for i, (img, info) in enumerate(zip(piece_images, piece_info)):
        row = i // cols
        col = i % cols
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(info, fontsize=8)
        axes[row, col].axis('off')
    
    plt.suptitle(f'All Squares from IMG_4762 (8x8 Grid)\nTotal: {n_pieces} squares', fontsize=16)
    plt.tight_layout()
    
    # Save the combined image
    combined_path = os.path.join(output_dir, "all_squares_grid.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"💾 Combined squares grid saved to: {combined_path}")
    
    # Show the plot
    plt.show()
    
    print(f"\n🔍 Review each square image:")
    print(f"   1. Check if the piece type matches the label")
    print(f"   2. Verify the square coordinates are correct")
    print(f"   3. Pay special attention to square b8 (should be empty)")
    print(f"   4. Look for any squares that seem to contain pieces from adjacent squares")

def save_annotation(corners, fen):
    """Save the corrected annotation."""
    annotation = {
        "image": "IMG_4762.JPG",
        "corners": corners,
        "fen": fen,
        "timestamp": "corrected_debug"
    }
    
    annotation_path = "grey_background_dataset/annotations/train/IMG_4762.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"💾 Corrected annotation saved to: {annotation_path}")

def main():
    """Main function to re-extract pieces from IMG_4762."""
    print("🔄 Re-extracting pieces from IMG_4762 with corrected FEN and corners")
    print("=" * 70)
    
    # Use the corrected corners and FEN
    corners = [
        [802, 2184],   # a8 (top-left)
        [2604, 2110],  # h8 (top-right)
        [2697, 4108],  # h1 (bottom-right)
        [473, 4020]    # a1 (bottom-left)
    ]
    
    fen = "rnbqk2r/1ppp1ppp/5n2/2b1p3/2BPP3/2N2N2/1PP2PPP/R1BQK2R w KQkq - 0 1"
    
    print(f"✅ Using corrected corners: {corners}")
    print(f"✅ Using FEN: {fen}")
    print()
    
    image_path = "grey_background_dataset/images/train/IMG_4762.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Save corrected annotation
        save_annotation(corners, fen)
        
        # Extract and show all pieces
        print("🔄 Extracting pieces...")
        pieces_extracted = extract_and_show_pieces(img, corners, fen, "IMG_4762")
        
        print(f"\n✅ Re-extraction complete!")
        print(f"♟️  Total pieces extracted: {pieces_extracted}")
        print(f"📁 All squares saved to: re_extracted_IMG_4762/")
        print(f"📁 Combined grid: re_extracted_IMG_4762/all_squares_grid.png")
        
        # Highlight the problematic square
        print(f"\n⚠️  Key square to verify:")
        print(f"   Square b8: Should be EMPTY according to FEN")
        print(f"   If this shows a piece, the corners or FEN need adjustment")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

