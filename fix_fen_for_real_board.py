#!/usr/bin/env python3
"""
Fix the FEN to match the actual piece positions on the board.
"""

import cv2
import numpy as np
import os
import chess

def analyze_current_fen():
    """Analyze the current FEN and show what it represents."""
    print("🔍 Analyzing current FEN...")
    
    current_fen = "3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1"
    
    print(f"📝 Current FEN: {current_fen}")
    
    # Parse current FEN
    board = chess.Board(current_fen)
    
    print(f"\n🎯 What the current FEN shows:")
    print("   Rank 8: 3r1r2 (3 empty, rook, 1 empty, rook, 2 empty)")
    print("   Rank 7: 3b2pk (3 empty, bishop, 2 empty, pawn, king)")
    print("   Rank 6: 1p1b2q1 (1 empty, pawn, 1 empty, bishop, 2 empty, queen, 1 empty)")
    print("   Rank 5: p1pN1p1p (pawn, 1 empty, pawn, knight, 1 empty, pawn, 1 empty, pawn)")
    print("   Rank 4: Q1P1n1pP (queen, 1 empty, pawn, 1 empty, knight, 1 empty, pawn, pawn)")
    print("   Rank 3: 1P1P1n2 (1 empty, pawn, 1 empty, pawn, 1 empty, knight, 2 empty)")
    print("   Rank 2: 1B4N1 (1 empty, bishop, 4 empty, knight, 1 empty)")
    print("   Rank 1: 1K1R1R1B (1 empty, king, 1 empty, rook, 1 empty, rook, 1 empty, bishop)")
    
    print(f"\n🔍 Key discrepancies you found:")
    print(f"   - h7: FEN shows king (k), but you see: EMPTY")
    print(f"   - e8: FEN shows empty, but you see: BLACK KING")
    print(f"   - a4: FEN shows queen (Q), you see: WHITE QUEEN ✓")

def get_correct_fen():
    """Get the correct FEN from user input."""
    print(f"\n📝 Please enter the CORRECT FEN for this position.")
    print(f"   Base it on what you actually see on the board, not the current FEN.")
    print(f"   Remember: FEN reads from rank 8 (top) to rank 1 (bottom)")
    print(f"   Each rank reads from a (left) to h (right)")
    
    print(f"\n🔍 Key positions to verify:")
    print(f"   - e8: Should contain black king (k)")
    print(f"   - h7: Should be empty (.)")
    print(f"   - a4: Should contain white queen (Q)")
    print(f"   - d8, f8: Should contain black rooks (r)")
    
    user_fen = input("\nEnter the correct FEN: ").strip()
    
    if user_fen:
        # Validate the FEN
        try:
            board = chess.Board(user_fen)
            print(f"✅ Valid FEN: {user_fen}")
            return user_fen
        except Exception as e:
            print(f"❌ Invalid FEN: {e}")
            print(f"   Please try again...")
            return get_correct_fen()
    else:
        print(f"❌ No FEN entered. Please provide a valid FEN.")
        return get_correct_fen()

def regenerate_pieces_with_correct_fen(correct_fen):
    """Regenerate piece images with the correct FEN."""
    print(f"\n🔄 Regenerating pieces with correct FEN...")
    
    # Image path
    image_path = "grey_background_dataset/images/test/NEW_20250805_135338_002.JPG"
    
    # Current corners (these are correct)
    corners = [
        [536, 1894],   # a8 (top-left)
        [2726, 1818],  # h8 (top-right)
        [2866, 4130],  # h1 (bottom-right)
        [359, 4101]    # a1 (bottom-left)
    ]
    
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
    
    # Parse correct FEN
    board = chess.Board(correct_fen)
    
    # Piece type mapping
    piece_mapping = {
        'P': 'white_pawn', 'R': 'white_rook', 'N': 'white_knight',
        'B': 'white_bishop', 'Q': 'white_queen', 'K': 'white_king',
        'p': 'black_pawn', 'r': 'black_rook', 'n': 'black_knight',
        'b': 'black_bishop', 'q': 'black_queen', 'k': 'black_king'
    }
    
    # Create output directory
    output_dir = "re_extracted_NEW_20250805_135338_002_corrected"
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

def main():
    """Main function to fix the FEN and regenerate pieces."""
    print("🔧 Fix FEN and Regenerate Pieces for NEW_20250805_135338_002")
    print("=" * 70)
    
    try:
        # Step 1: Analyze current FEN
        analyze_current_fen()
        
        # Step 2: Get correct FEN from user
        correct_fen = get_correct_fen()
        
        # Step 3: Regenerate pieces with correct FEN
        output_dir = regenerate_pieces_with_correct_fen(correct_fen)
        
        print(f"\n✅ Fix complete!")
        print(f"📝 Corrected FEN: {correct_fen}")
        print(f"🖼️  Pieces regenerated to: {output_dir}")
        
        print(f"\n🔍 Next steps:")
        print(f"   1. Review the new pieces in: {output_dir}")
        print(f"   2. Verify they match what you see on the board")
        print(f"   3. If satisfied, replace the dataset pieces")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
