#!/usr/bin/env python3
"""
Fix misplaced pieces in white_king directory based on user feedback.
"""

import os
import shutil
import json
from pathlib import Path

def get_fen_for_image(image_name, dataset_type):
    """Get FEN from annotation file."""
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            return data.get('fen', '')
    return ''

def get_piece_at_square(fen, square):
    """Get piece at specific square from FEN."""
    if not fen:
        return None
    
    # Parse FEN (first part is board position)
    board_fen = fen.split()[0]
    
    # Convert square notation to rank/file
    file_letter = square[0]  # a-h
    rank_num = int(square[1])  # 1-8
    
    # Convert to 0-based indices
    file_idx = ord(file_letter) - ord('a')  # 0-7
    rank_idx = 8 - rank_num  # 0-7 (FEN ranks are reversed)
    
    # Parse FEN board
    board = []
    for char in board_fen:
        if char == '/':
            continue
        elif char.isdigit():
            # Empty squares
            board.extend([''] * int(char))
        else:
            board.append(char)
    
    if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
        piece = board[rank_idx * 8 + file_idx]
        return piece if piece else None
    
    return None

def main():
    # User-reported issues in white_king directory
    issues = [
        # (image_name, square, issue_type, correct_piece)
        ("IMG_4785", "e4", "blank", None),
        ("NEW_20250805_135338_000", "b2", "wrong_piece", "B"),  # white bishop
        ("NEW_20250805_135338_001", "b3", "wrong_piece", "P"),  # white pawn
        ("NEW_20250805_135338_002", "b1", "blank", None),
        ("NEW_20250805_135338_003", "g2", "wrong_piece", "B"),  # white bishop
        ("NEW_20250805_135338_004", "b3", "blank", None),
        ("NEW_20250805_135338_005", "g2", "wrong_piece", "B"),  # white bishop
        ("NEW_20250805_135338_006", "b3", "wrong_piece", "P"),  # white pawn
        ("NEW_20250805_135338_007", "f1", "wrong_piece", "R"),  # white rook
        ("NEW_20250805_135338_009", "g1", "blank", None),
        ("NEW_20250805_135338_012", "b2", "blank", None),
        ("NEW_20250805_135338_013", "b3", "blank", None),
    ]
    
    white_king_dir = "grey_background_dataset/pieces/test/white_king"
    removed_count = 0
    moved_count = 0
    
    print("🔍 Fixing misplaced pieces in white_king directory...")
    
    for image_name, square, issue_type, correct_piece in issues:
        piece_file = f"{image_name}_{square}.png"
        piece_path = os.path.join(white_king_dir, piece_file)
        
        if not os.path.exists(piece_path):
            print(f"⚠️  Piece file not found: {piece_file}")
            continue
        
        # Determine dataset type
        dataset_type = "test"  # All pieces are in test directory
        
        # Get FEN to verify
        fen = get_fen_for_image(image_name, dataset_type)
        if fen:
            actual_piece = get_piece_at_square(fen, square)
            print(f"📋 {piece_file}: FEN shows '{actual_piece}' at {square}")
        
        if issue_type == "blank":
            # Remove blank square
            os.remove(piece_path)
            print(f"🗑️  Removed blank square: {piece_file}")
            removed_count += 1
            
        elif issue_type == "wrong_piece":
            # Move to correct directory
            if correct_piece == "B":  # white bishop
                target_dir = "grey_background_dataset/pieces/test/white_bishop"
            elif correct_piece == "P":  # white pawn
                target_dir = "grey_background_dataset/pieces/test/white_pawn"
            elif correct_piece == "R":  # white rook
                target_dir = "grey_background_dataset/pieces/test/white_rook"
            else:
                print(f"⚠️  Unknown piece type: {correct_piece}")
                continue
            
            target_path = os.path.join(target_dir, piece_file)
            shutil.move(piece_path, target_path)
            print(f"📦 Moved {piece_file} to {correct_piece} directory")
            moved_count += 1
    
    print(f"\n✅ White King cleanup complete:")
    print(f"   - Removed {removed_count} blank squares")
    print(f"   - Moved {moved_count} misplaced pieces")

if __name__ == "__main__":
    main()
