#!/usr/bin/env python3
"""
Comprehensive cleanup of white_rook directory - check all pieces against FEN.
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

def get_piece_directory(piece):
    """Get target directory for piece type."""
    piece_dirs = {
        'K': 'white_king',
        'Q': 'white_queen', 
        'R': 'white_rook',
        'B': 'white_bishop',
        'N': 'white_knight',
        'P': 'white_pawn',
        'k': 'black_king',
        'q': 'black_queen',
        'r': 'black_rook', 
        'b': 'black_bishop',
        'n': 'black_knight',
        'p': 'black_pawn'
    }
    return piece_dirs.get(piece)

def main():
    white_rook_dir = "grey_background_dataset/pieces/test/white_rook"
    
    if not os.path.exists(white_rook_dir):
        print(f"❌ Directory not found: {white_rook_dir}")
        return
    
    pieces = [f for f in os.listdir(white_rook_dir) if f.endswith('.png')]
    print(f"🔍 Checking {len(pieces)} pieces in white_rook directory...")
    
    removed_count = 0
    moved_count = 0
    confirmed_count = 0
    
    for piece_file in pieces:
        # Parse filename: IMG_XXXX_square.png or NEW_XXXX_square.png
        parts = piece_file.replace('.png', '').split('_')
        if len(parts) >= 3:
            image_name = '_'.join(parts[:-1])
            square = parts[-1]
        else:
            print(f"⚠️  Cannot parse filename: {piece_file}")
            continue
        
        piece_path = os.path.join(white_rook_dir, piece_file)
        
        # Determine dataset type
        dataset_type = "test"  # All pieces are in test directory
        
        # Get FEN to verify
        fen = get_fen_for_image(image_name, dataset_type)
        if not fen:
            print(f"⚠️  No FEN found for {image_name}")
            continue
        
        actual_piece = get_piece_at_square(fen, square)
        
        if actual_piece is None:
            # Empty square - remove
            os.remove(piece_path)
            print(f"🗑️  Removed empty square: {piece_file}")
            removed_count += 1
            
        elif actual_piece == 'R':
            # Correct white rook - keep
            confirmed_count += 1
            
        else:
            # Wrong piece - move to correct directory
            target_dir_name = get_piece_directory(actual_piece)
            if target_dir_name:
                target_dir = f"grey_background_dataset/pieces/test/{target_dir_name}"
                target_path = os.path.join(target_dir, piece_file)
                
                if os.path.exists(target_dir):
                    shutil.move(piece_path, target_path)
                    print(f"📦 Moved {piece_file} to {target_dir_name} (was {actual_piece})")
                    moved_count += 1
                else:
                    print(f"⚠️  Target directory not found: {target_dir}")
            else:
                print(f"⚠️  Unknown piece type: {actual_piece}")
    
    print(f"\n✅ Comprehensive white_rook cleanup complete:")
    print(f"   - Confirmed {confirmed_count} correct white rooks")
    print(f"   - Removed {removed_count} empty squares")
    print(f"   - Moved {moved_count} misplaced pieces")

if __name__ == "__main__":
    main()
