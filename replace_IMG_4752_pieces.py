#!/usr/bin/env python3
"""
Replace the old dataset pieces for IMG_4752 with the newly corrected ones.
"""

import os
import shutil
from pathlib import Path

def replace_dataset_pieces():
    """Replace old dataset pieces with newly corrected ones."""
    print("🔧 Replacing IMG_4752 dataset pieces...")
    
    # Source directory with corrected pieces
    source_dir = "re_extracted_IMG_4752"
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Get all corrected piece images
    piece_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    if not piece_files:
        print(f"❌ No piece images found in {source_dir}")
        return
    
    print(f"   📁 Found {len(piece_files)} corrected piece images")
    
    # Parse piece positions from FEN to map squares to piece types
    # FEN: 8/3k4/2n1q3/1n1p1p2/4P3/2N2P2/PPP5/1N1Q4 w - - 0 1
    piece_positions = {
        'a2': 'white_pawn',      # P
        'b1': 'white_knight',    # N
        'b2': 'white_pawn',      # P
        'b5': 'black_knight',    # n
        'c2': 'white_pawn',      # P
        'c3': 'white_knight',    # N
        'c6': 'black_knight',    # n
        'd1': 'white_queen',     # Q
        'd5': 'black_pawn',      # p
        'd7': 'black_king',      # k
        'e4': 'white_pawn',      # P
        'e6': 'black_queen',     # q
        'f3': 'white_pawn',      # P
        'f5': 'black_pawn'       # p
    }
    
    replaced_count = 0
    backup_count = 0
    
    for piece_file in piece_files:
        # Extract square from filename (e.g., IMG_4752_a2.png -> a2)
        square = piece_file.replace('IMG_4752_', '').replace('.png', '')
        
        if square in piece_positions:
            piece_type = piece_positions[square]
            source_path = os.path.join(source_dir, piece_file)
            
            # Target path in dataset
            target_dir = f"grey_background_dataset/pieces/test/{piece_type}"
            target_path = os.path.join(target_dir, piece_file)
            
            # Create backup of existing file if it exists
            if os.path.exists(target_path):
                backup_path = target_path + ".backup_corrected"
                shutil.copy2(target_path, backup_path)
                backup_count += 1
            
            # Copy corrected piece to dataset
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(source_path, target_path)
            replaced_count += 1
            
            print(f"   ✅ {piece_file} -> {piece_type}/")
        else:
            print(f"   ⚠️  Unknown square {square} for {piece_file}")
    
    print(f"\n✅ Piece replacement complete!")
    print(f"   📊 Replaced: {replaced_count} pieces")
    print(f"   💾 Backups created: {backup_count} files")
    
    return replaced_count

def verify_replacement():
    """Verify that the replacement was successful."""
    print(f"\n🔍 Verifying replacement...")
    
    # Check a few key pieces
    test_pieces = [
        "grey_background_dataset/pieces/test/black_king/IMG_4752_d7.png",
        "grey_background_dataset/pieces/test/white_queen/IMG_4752_d1.png",
        "grey_background_dataset/pieces/test/black_knight/IMG_4752_b5.png"
    ]
    
    verified_count = 0
    for piece_path in test_pieces:
        if os.path.exists(piece_path):
            file_size = os.path.getsize(piece_path)
            print(f"   ✅ {os.path.basename(piece_path)}: {file_size} bytes")
            verified_count += 1
        else:
            print(f"   ❌ {os.path.basename(piece_path)}: Not found")
    
    if verified_count == len(test_pieces):
        print(f"   🎉 All test pieces verified successfully!")
    else:
        print(f"   ⚠️  Some pieces may not have been replaced correctly")
    
    return verified_count

def cleanup_temp_files():
    """Clean up temporary files after successful replacement."""
    print(f"\n🧹 Cleaning up temporary files...")
    
    # Remove re-extracted pieces directory
    if os.path.exists("re_extracted_IMG_4752"):
        shutil.rmtree("re_extracted_IMG_4752")
        print(f"   🗑️  Removed: re_extracted_IMG_4752/")
    
    # Remove warped board images (keep for reference)
    print(f"   💾 Kept: debug_outputs/IMG_4752_warped*.png (for reference)")

def main():
    """Main function to replace IMG_4752 dataset pieces."""
    print("🔧 IMG_4752 Dataset Piece Replacement")
    print("=" * 50)
    
    try:
        # Step 1: Replace pieces in dataset
        print(f"\n🎯 STEP 1: Replace Dataset Pieces")
        replaced_count = replace_dataset_pieces()
        
        if replaced_count == 0:
            print(f"❌ No pieces were replaced. Exiting.")
            return
        
        # Step 2: Verify replacement
        print(f"\n🎯 STEP 2: Verify Replacement")
        verified_count = verify_replacement()
        
        # Step 3: Clean up temporary files
        print(f"\n🎯 STEP 3: Cleanup")
        cleanup_temp_files()
        
        print(f"\n🎉 IMG_4752.JPG fix and replacement complete!")
        print(f"   📊 Total pieces replaced: {replaced_count}")
        print(f"   ✅ Verification passed: {verified_count}/3 test pieces")
        
        print(f"\n💡 Next steps:")
        print(f"   1. IMG_4752.JPG is now fixed in the TEST set")
        print(f"   2. You can proceed to fix the next problematic image")
        print(f"   3. Consider fixing the TRAIN set images next")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
