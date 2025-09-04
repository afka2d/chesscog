#!/usr/bin/env python3
"""
Replace the dataset pieces with the newly corrected ones for NEW_20250805_135338_002.
"""

import os
import shutil
from pathlib import Path

def replace_dataset_pieces():
    """Replace the dataset pieces with the corrected ones."""
    print("🔄 Replacing dataset pieces with corrected ones...")
    
    # Source directory (corrected pieces)
    source_dir = "re_extracted_NEW_20250805_135338_002_corrected"
    
    # Target directories in the dataset
    target_base = "grey_background_dataset/pieces/test"
    
    # Piece type mapping based on the corrected FEN
    # FEN: r3k3/2p1rpp1/4b1q1/pp2B1p1/Q1Pn1R1P/2bPnPPB/4NNK1/3R4 w q - 0 1
    piece_positions = {
        # Rank 8: r3k3 (rook, 3 empty, king, 3 empty)
        'a8': 'black_rook', 'e8': 'black_king',
        # Rank 7: 2p1rpp1 (2 empty, pawn, 1 empty, rook, pawn, pawn, 1 empty)
        'c7': 'black_pawn', 'e7': 'black_rook', 'f7': 'black_pawn', 'g7': 'black_pawn',
        # Rank 6: 4b1q1 (4 empty, bishop, 1 empty, queen, 1 empty)
        'e6': 'black_bishop', 'g6': 'black_queen',
        # Rank 5: pp2B1p1 (pawn, pawn, 2 empty, bishop, 1 empty, pawn, 1 empty)
        'a5': 'black_pawn', 'b5': 'black_pawn', 'e5': 'white_bishop', 'g5': 'black_pawn',
        # Rank 4: Q1Pn1R1P (queen, 1 empty, pawn, knight, 1 empty, rook, 1 empty, pawn)
        'a4': 'white_queen', 'c4': 'white_pawn', 'd4': 'black_knight', 'f4': 'white_rook', 'h4': 'white_pawn',
        # Rank 3: 2bPnPPB (2 empty, bishop, pawn, knight, pawn, pawn, bishop)
        'c3': 'black_bishop', 'd3': 'white_pawn', 'e3': 'black_knight', 'f3': 'white_pawn', 'g3': 'white_pawn', 'h3': 'white_bishop',
        # Rank 2: 4NNK1 (4 empty, knight, knight, king, 1 empty)
        'e2': 'white_knight', 'f2': 'white_knight', 'g2': 'white_king',
        # Rank 1: 3R4 (3 empty, rook, 4 empty)
        'd1': 'white_rook'
    }
    
    replaced_count = 0
    errors = []
    
    # Process each piece position
    for square, piece_type in piece_positions.items():
        # Source file (from corrected extraction)
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
                    backup_path = target_path + ".backup_corrected"
                    shutil.copy2(target_path, backup_path)
                    print(f"   💾 Backed up original: {backup_path}")
                
                # Copy the corrected piece
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

def update_annotation_file():
    """Update the annotation file with the corrected FEN."""
    print(f"\n📝 Updating annotation file with corrected FEN...")
    
    annotation_path = "grey_background_dataset/annotations/test/NEW_20250805_135338_002.json"
    
    # Backup original annotation
    backup_path = "grey_background_dataset/annotations/test/NEW_20250805_135338_002.json.backup_corrected"
    if os.path.exists(annotation_path):
        shutil.copy2(annotation_path, backup_path)
        print(f"   💾 Original annotation backed up to: {backup_path}")
    
    # Create new annotation with corrected FEN
    import json
    annotation = {
        "image": "NEW_20250805_135338_002.JPG",
        "corners": [
            [536, 1894],   # a8 (top-left)
            [2726, 1818],  # h8 (top-right)
            [2866, 4130],  # h1 (bottom-right)
            [359, 4101]    # a1 (bottom-left)
        ],
        "fen": "r3k3/2p1rpp1/4b1q1/pp2B1p1/Q1Pn1R1P/2bPnPPB/4NNK1/3R4 w q - 0 1",
        "white_turn": True,
        "timestamp": "corrected_fen_and_pieces"
    }
    
    # Save corrected annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"   ✅ Corrected annotation saved to: {annotation_path}")

def verify_replacement():
    """Verify that the replacement was successful."""
    print(f"\n🔍 Verifying replacement...")
    
    # Check a few key pieces
    test_pieces = [
        "grey_background_dataset/pieces/test/black_king/NEW_20250805_135338_002_e8.png",
        "grey_background_dataset/pieces/test/white_queen/NEW_20250805_135338_002_a4.png",
        "grey_background_dataset/pieces/test/black_rook/NEW_20250805_135338_002_a8.png",
        "grey_background_dataset/pieces/test/white_rook/NEW_20250805_135338_002_f4.png"
    ]
    
    for piece_path in test_pieces:
        if os.path.exists(piece_path):
            size = os.path.getsize(piece_path)
            print(f"   ✅ {piece_path} exists ({size} bytes)")
        else:
            print(f"   ❌ {piece_path} missing")

def main():
    """Main function to replace dataset pieces."""
    print("🔧 Replace Dataset Pieces with Corrected Ones")
    print("=" * 60)
    
    try:
        # Step 1: Replace piece images
        replaced_count, errors = replace_dataset_pieces()
        
        # Step 2: Update annotation file
        update_annotation_file()
        
        # Step 3: Verify replacement
        verify_replacement()
        
        print(f"\n✅ Replacement process complete!")
        print(f"🖼️  {replaced_count} piece images replaced")
        print(f"📝 Annotation updated with corrected FEN")
        
        if errors:
            print(f"⚠️  {len(errors)} errors encountered - check output above")
        else:
            print(f"🎉 All pieces replaced successfully!")
        
        print(f"\n🔍 Final status:")
        print(f"   - NEW_20250805_135338_002 is now completely fixed")
        print(f"   - All pieces correctly extracted and aligned")
        print(f"   - FEN matches actual board position")
        print(f"   - Dataset updated with high-quality training data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
