#!/usr/bin/env python3
"""
Comprehensive cleanup of the black_queen directory.
Check all pieces against their FENs and remove/move any that don't belong.
"""

import os
import json
import chess
import shutil

def check_piece_against_fen(image_name, square, dataset_type="test"):
    """Check if a piece is correctly placed according to its FEN."""
    try:
        annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
        if not os.path.exists(annotation_path):
            return "no_annotation", None
        
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        fen = annotation['fen']
        board = chess.Board(fen)
        square_obj = chess.parse_square(square)
        piece = board.piece_at(square_obj)
        
        if piece is None:
            return "empty", None
        
        # Determine expected piece type
        color = 'white' if piece.color else 'black'
        piece_symbol = piece.symbol().lower()
        
        if piece_symbol == 'p':
            expected_type = f"{color}_pawn"
        elif piece_symbol == 'r':
            expected_type = f"{color}_rook"
        elif piece_symbol == 'n':
            expected_type = f"{color}_knight"
        elif piece_symbol == 'b':
            expected_type = f"{color}_bishop"
        elif piece_symbol == 'q':
            expected_type = f"{color}_queen"
        elif piece_symbol == 'k':
            expected_type = f"{color}_king"
        else:
            expected_type = f"{color}_{piece_symbol}"
        
        return "correct", expected_type
        
    except Exception as e:
        return "error", str(e)

def main():
    """Comprehensive cleanup of black_queen directory."""
    print("🔧 COMPREHENSIVE BLACK QUEEN DIRECTORY CLEANUP")
    print("=" * 60)
    print("Checking all pieces against their FENs...")
    print()
    
    black_queen_dir = "grey_background_dataset/pieces/test/black_queen"
    
    if not os.path.exists(black_queen_dir):
        print(f"❌ Directory not found: {black_queen_dir}")
        return
    
    # Get all pieces in black_queen directory
    pieces = [f for f in os.listdir(black_queen_dir) if f.endswith('.png')]
    print(f"📊 Found {len(pieces)} pieces in black_queen directory")
    print()
    
    to_remove = []
    to_move = {}
    correct_pieces = []
    errors = []
    
    for piece_file in pieces:
        # Extract image name and square from filename
        # Format: NEW_20250805_135338_XXX_square.png
        parts = piece_file.split('_')
        if len(parts) >= 5:
            image_name = '_'.join(parts[:-1])  # Everything except the last part (square.png)
            square = parts[-1].replace('.png', '')
            
            status, expected_type = check_piece_against_fen(image_name, square)
            
            if status == "empty":
                to_remove.append(piece_file)
                print(f"🗑️  {piece_file}: Should be removed (FEN says empty)")
            elif status == "correct":
                if expected_type == "black_queen":
                    correct_pieces.append(piece_file)
                    print(f"✅ {piece_file}: Correctly placed")
                else:
                    to_move[piece_file] = expected_type
                    print(f"📁 {piece_file}: Should move to {expected_type}/")
            elif status == "no_annotation":
                errors.append(f"{piece_file}: No annotation found")
                print(f"❌ {piece_file}: No annotation found")
            elif status == "error":
                errors.append(f"{piece_file}: {expected_type}")
                print(f"❌ {piece_file}: Error - {expected_type}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Correctly placed: {len(correct_pieces)}")
    print(f"   🗑️  To remove: {len(to_remove)}")
    print(f"   📁 To move: {len(to_move)}")
    print(f"   ❌ Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ ERRORS:")
        for error in errors:
            print(f"   - {error}")
    
    # Perform cleanup
    removed_count = 0
    moved_count = 0
    
    print(f"\n🔧 PERFORMING CLEANUP:")
    
    # Remove pieces that should be empty
    for piece_file in to_remove:
        piece_path = os.path.join(black_queen_dir, piece_file)
        os.remove(piece_path)
        removed_count += 1
        print(f"   🗑️  Removed: {piece_file}")
    
    # Move pieces to correct directories
    for piece_file, target_type in to_move.items():
        source_path = os.path.join(black_queen_dir, piece_file)
        target_path = f"grey_background_dataset/pieces/test/{target_type}/{piece_file}"
        target_dir = os.path.dirname(target_path)
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Move the file
        shutil.move(source_path, target_path)
        moved_count += 1
        print(f"   📁 Moved: {piece_file} -> {target_type}/")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"   🗑️  Removed: {removed_count} pieces")
    print(f"   📁 Moved: {moved_count} pieces")
    print(f"   ✅ Correctly placed: {len(correct_pieces)} pieces")
    
    # Final count
    remaining_pieces = [f for f in os.listdir(black_queen_dir) if f.endswith('.png')]
    print(f"   📊 Remaining in black_queen: {len(remaining_pieces)} pieces")

if __name__ == "__main__":
    main()
