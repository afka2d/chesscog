#!/usr/bin/env python3
"""
Replace pieces for NEW_20250805_135338_013 with the newly extracted ones.
"""

import os
import json
import chess
import shutil

def main():
    """Replace pieces for NEW_20250805_135338_013."""
    print("🔧 REPLACING PIECES FOR NEW_20250805_135338_013")
    print("=" * 50)
    
    image_name = "NEW_20250805_135338_013"
    dataset_type = "test"
    extract_dir = "re_extracted_NEW_20250805_135338_013"
    
    # Load the new annotation to get the FEN
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    fen = annotation['fen']
    print(f"📝 Using FEN: {fen}")
    
    # Parse FEN to get piece positions
    board = chess.Board(fen)
    piece_positions = {}
    
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            piece = board.piece_at(square)
            
            if piece is not None:
                square_name = chess.square_name(square)
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
                
                piece_positions[square_name] = piece_type
    
    print(f"📊 Found {len(piece_positions)} pieces in FEN")
    
    # Replace pieces in dataset
    replaced_count = 0
    backup_count = 0
    
    for piece_file in os.listdir(extract_dir):
        if piece_file.endswith('.png'):
            # Extract square name from filename
            square_name = piece_file.split('_')[-1].replace('.png', '')
            
            if square_name in piece_positions:
                piece_type = piece_positions[square_name]
                source_path = os.path.join(extract_dir, piece_file)
                target_path = f"grey_background_dataset/pieces/{dataset_type}/{piece_type}/{piece_file}"
                
                # Create backup if target exists
                if os.path.exists(target_path):
                    backup_path = target_path + ".backup_manual_fix"
                    os.rename(target_path, backup_path)
                    backup_count += 1
                
                # Copy new piece
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(source_path, target_path)
                replaced_count += 1
                print(f"   ✅ {piece_file} -> {piece_type}/")
            else:
                print(f"   ⚠️  Unknown square {square_name} for {piece_file}")
    
    print(f"   📊 Replaced: {replaced_count} pieces")
    print(f"   💾 Backups created: {backup_count} files")
    
    # Cleanup
    print(f"🧹 Cleaning up temporary files...")
    shutil.rmtree(extract_dir)
    print(f"   🗑️  Removed: {extract_dir}/")
    
    print(f"\n🎉 NEW_20250805_135338_013 pieces replacement complete!")
    print(f"   📊 Total pieces replaced: {replaced_count}")

if __name__ == "__main__":
    main()
