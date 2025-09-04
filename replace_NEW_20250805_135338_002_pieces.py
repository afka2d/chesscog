#!/usr/bin/env python3
"""
Replace the original distorted piece images for NEW_20250805_135338_002 with the correctly extracted ones.
"""

import os
import shutil
from pathlib import Path

def replace_piece_images():
    """Replace the original piece images with the correctly extracted ones."""
    print("🔄 Replacing original piece images for NEW_20250805_135338_002...")
    
    # Source directory (correctly extracted pieces)
    source_dir = "re_extracted_NEW_20250805_135338_002"
    
    # Target directories in the dataset
    target_base = "grey_background_dataset/pieces/test"
    
    # Piece type mapping based on the FEN
    # FEN: 3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1
    piece_positions = {
        # Rank 8: 3r1r2 (3 empty, r, 1 empty, r, 2 empty)
        'd8': 'black_rook', 'f8': 'black_rook',
        # Rank 7: 3b2pk (3 empty, b, 2 empty, p, k)
        'd7': 'black_bishop', 'g7': 'black_pawn', 'h7': 'black_king',
        # Rank 6: 1p1b2q1 (1 empty, p, 1 empty, b, 2 empty, q, 1 empty)
        'b6': 'black_pawn', 'd6': 'black_bishop', 'g6': 'black_queen',
        # Rank 5: p1pN1p1p (p, 1 empty, p, N, 1 empty, p, 1 empty, p)
        'a5': 'black_pawn', 'c5': 'black_pawn', 'd5': 'white_knight', 'f5': 'black_pawn', 'h5': 'black_pawn',
        # Rank 4: Q1P1n1pP (Q, 1 empty, P, 1 empty, n, 1 empty, p, P)
        'a4': 'white_queen', 'c4': 'white_pawn', 'e4': 'black_knight', 'g4': 'black_pawn', 'h4': 'white_pawn',
        # Rank 3: 1P1P1n2 (1 empty, P, 1 empty, P, 1 empty, n, 2 empty)
        'b3': 'white_pawn', 'd3': 'white_pawn', 'f3': 'black_knight',
        # Rank 2: 1B4N1 (1 empty, B, 4 empty, N, 1 empty)
        'b2': 'white_bishop', 'g2': 'white_knight',
        # Rank 1: 1K1R1R1B (1 empty, K, 1 empty, R, 1 empty, R, 1 empty, B)
        'b1': 'white_king', 'd1': 'white_rook', 'f1': 'white_rook', 'h1': 'white_bishop'
    }
    
    replaced_count = 0
    errors = []
    
    # Process each piece position
    for square, piece_type in piece_positions.items():
        # Source file (from re-extraction)
        source_file = f"NEW_20250805_135338_002_{square}.png"
        source_path = os.path.join(source_dir, source_file)
        
        # Target file (in dataset)
        target_dir = os.path.join(target_base, piece_type)
        target_path = os.path.join(target_dir, source_file)
        
        if os.path.exists(source_path):
            try:
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Backup original file if it exists
                if os.path.exists(target_path):
                    backup_path = target_path + ".backup"
                    shutil.copy2(target_path, backup_path)
                    print(f"   💾 Backed up original: {backup_path}")
                
                # Copy the correctly extracted piece
                shutil.copy2(source_path, target_path)
                print(f"   ✅ Replaced {piece_type} from {square} -> {piece_type}/{source_file}")
                replaced_count += 1
                
            except Exception as e:
                error_msg = f"Error replacing {square}: {e}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
        else:
            error_msg = f"Source file not found: {source_path}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Successfully replaced: {replaced_count} pieces")
    print(f"   ❌ Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors:
            print(f"   - {error}")
    
    return replaced_count, errors

def verify_replacement():
    """Verify that the replacement was successful."""
    print(f"\n🔍 Verifying replacement...")
    
    # Check a few key pieces
    test_pieces = [
        "grey_background_dataset/pieces/test/black_rook/NEW_20250805_135338_002_d8.png",
        "grey_background_dataset/pieces/test/black_king/NEW_20250805_135338_002_h7.png",
        "grey_background_dataset/pieces/test/white_queen/NEW_20250805_135338_002_a4.png"
    ]
    
    for piece_path in test_pieces:
        if os.path.exists(piece_path):
            size = os.path.getsize(piece_path)
            print(f"   ✅ {piece_path} exists ({size} bytes)")
        else:
            print(f"   ❌ {piece_path} missing")
    
    # Check if the original distorted piece still exists
    original_distorted = "grey_background_dataset/pieces/test/black_rook/NEW_20250805_135337_002_a8.png"
    if os.path.exists(original_distorted):
        print(f"   ⚠️  Original distorted piece still exists: {original_distorted}")
        print(f"      You may want to remove this manually if it's no longer needed")
    else:
        print(f"   ✅ Original distorted piece has been replaced")

def main():
    """Main function to replace piece images."""
    print("🔧 Replacing Piece Images for NEW_20250805_135338_002")
    print("=" * 60)
    
    try:
        # Step 1: Replace piece images
        replaced_count, errors = replace_piece_images()
        
        # Step 2: Verify replacement
        verify_replacement()
        
        print(f"\n✅ Replacement process complete!")
        print(f"🖼️  {replaced_count} piece images replaced")
        
        if errors:
            print(f"⚠️  {len(errors)} errors encountered - check output above")
        else:
            print(f"🎉 All pieces replaced successfully!")
        
        print(f"\n🔍 Next steps:")
        print(f"   1. Review the replaced pieces in the dataset")
        print(f"   2. Test the piece classifier with the corrected data")
        print(f"   3. Consider applying similar fixes to other distorted images")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
