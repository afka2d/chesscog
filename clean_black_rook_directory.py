#!/usr/bin/env python3
"""
Clean up all misplaced pieces in the black_rook directory.
This script will move pieces to their correct directories or remove them if they're empty squares.
"""

import os
import json
import chess
import shutil

def get_correct_piece_type(fen, square_name):
    """Get the correct piece type for a square according to the FEN."""
    try:
        board = chess.Board(fen)
        square_obj = chess.parse_square(square_name)
        piece = board.piece_at(square_obj)
        
        if piece is None:
            return None  # Empty square
        
        color = 'white' if piece.color else 'black'
        piece_symbol = piece.symbol().lower()
        
        if piece_symbol == 'p':
            return f'{color}_pawn'
        elif piece_symbol == 'r':
            return f'{color}_rook'
        elif piece_symbol == 'n':
            return f'{color}_knight'
        elif piece_symbol == 'b':
            return f'{color}_bishop'
        elif piece_symbol == 'q':
            return f'{color}_queen'
        elif piece_symbol == 'k':
            return f'{color}_king'
        else:
            return f'{color}_{piece_symbol}'
    except:
        return None

def main():
    """Clean up all misplaced pieces in the black_rook directory."""
    print("🔧 CLEANING BLACK ROOK DIRECTORY")
    print("=" * 50)
    print("Moving all misplaced pieces to their correct directories.")
    print()
    
    black_rook_dir = 'grey_background_dataset/pieces/test/black_rook'
    if not os.path.exists(black_rook_dir):
        print("❌ Black rook directory not found")
        return
    
    pieces = [f for f in os.listdir(black_rook_dir) if f.endswith('.png')]
    print(f"📊 Found {len(pieces)} pieces in black_rook directory")
    
    moved_count = 0
    removed_count = 0
    correct_count = 0
    
    for piece_file in pieces:
        # Extract image name and square
        parts = piece_file.split('_')
        if len(parts) < 4:
            print(f"⚠️  {piece_file}: Cannot parse filename")
            continue
        
        image_name = '_'.join(parts[:-1])
        square = parts[-1].replace('.png', '')
        
        try:
            # Load annotation
            annotation_path = f'grey_background_dataset/annotations/test/{image_name}.json'
            if not os.path.exists(annotation_path):
                print(f"⚠️  {piece_file}: No annotation found")
                continue
            
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            fen = annotation['fen']
            correct_piece_type = get_correct_piece_type(fen, square)
            
            if correct_piece_type is None:
                # Empty square - remove the piece
                source_path = os.path.join(black_rook_dir, piece_file)
                os.remove(source_path)
                print(f"🗑️  {piece_file}: Removed (empty square)")
                removed_count += 1
                
            elif correct_piece_type == 'black_rook':
                # Already in correct location
                print(f"✅ {piece_file}: Correct (black rook)")
                correct_count += 1
                
            else:
                # Move to correct directory
                source_path = os.path.join(black_rook_dir, piece_file)
                target_path = f'grey_background_dataset/pieces/test/{correct_piece_type}/{piece_file}'
                target_dir = os.path.dirname(target_path)
                
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Move the file
                shutil.move(source_path, target_path)
                print(f"📁 {piece_file}: Moved to {correct_piece_type}/")
                moved_count += 1
                
        except Exception as e:
            print(f"❌ {piece_file}: Error processing - {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"📊 Results:")
    print(f"   ✅ Correct pieces: {correct_count}")
    print(f"   📁 Moved pieces: {moved_count}")
    print(f"   🗑️  Removed pieces: {removed_count}")
    print(f"   📊 Total processed: {correct_count + moved_count + removed_count}")

if __name__ == "__main__":
    main()
