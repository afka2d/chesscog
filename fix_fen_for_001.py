#!/usr/bin/env python3
"""
Fix the FEN for NEW_20250805_135338_001 to include the white bishop on f4.
"""

import os
import json
import chess

def main():
    """Fix the FEN for NEW_20250805_135338_001."""
    print("🔧 FIXING FEN FOR NEW_20250805_135338_001")
    print("=" * 50)
    print("The current FEN says f4 is empty, but you see a white bishop.")
    print("Let's correct the FEN to include the white bishop on f4.")
    print()
    
    image_name = "NEW_20250805_135338_001"
    dataset_type = "test"
    
    # Load current annotation
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    current_fen = annotation['fen']
    print(f"📝 Current FEN: {current_fen}")
    
    # Parse current FEN
    board = chess.Board(current_fen)
    
    # Set white bishop on f4
    f4_square = chess.parse_square('f4')
    board.set_piece_at(f4_square, chess.Piece(chess.BISHOP, chess.WHITE))
    
    new_fen = board.fen()
    print(f"📝 New FEN: {new_fen}")
    
    # Verify f4 has white bishop
    f4_piece = board.piece_at(f4_square)
    if f4_piece and f4_piece.symbol() == 'B':
        print(f"✅ f4 now contains white bishop")
    else:
        print(f"❌ Error: f4 does not contain white bishop")
        return
    
    # Update annotation
    annotation['fen'] = new_fen
    annotation['timestamp'] = 'fen_corrected_f4_bishop'
    
    # Create backup
    backup_path = annotation_path + ".backup_f4_fix"
    os.rename(annotation_path, backup_path)
    print(f"💾 Created backup: {backup_path}")
    
    # Save new annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"✅ Updated annotation: {annotation_path}")
    print(f"📝 New FEN: {new_fen}")
    
    print(f"\n🎉 FEN fix complete!")
    print(f"   f4 now correctly contains a white bishop")

if __name__ == "__main__":
    main()
