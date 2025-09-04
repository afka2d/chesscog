#!/usr/bin/env python3
"""
Fix misplaced pieces in the white_bishop directory.
Remove empty squares, move pieces to correct directories.
"""

import os
import shutil

def main():
    """Fix misplaced pieces in white_bishop directory."""
    print("🔧 FIXING WHITE BISHOP DIRECTORY MISPLACED PIECES")
    print("=" * 60)
    print("Based on FEN analysis, fixing misplaced pieces...")
    print()
    
    # Define the fixes needed based on FEN analysis
    fixes = [
        # (source_file, action, target_directory, reason)
        ("NEW_20250805_135338_000_e6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_000_g2.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_c4.png", "move", "white_pawn", "FEN says white_pawn, you see white_pawn"),
        ("NEW_20250805_135338_003_g6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_g2.png", "move", "white_knight", "FEN says white_knight, you see white_knight"),
        ("NEW_20250805_135338_002_b2.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_002_h1.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_003_b2.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_003_h3.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_004_c4.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_004_f2.png", "move", "white_pawn", "FEN says white_pawn, you see white_pawn"),
        ("NEW_20250805_135338_005_e5.png", "move", "black_bishop", "FEN says black_bishop, you see black_bishop"),
        ("NEW_20250805_135338_005_h3.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_006_c4.png", "move", "white_pawn", "FEN says white_pawn, you see white_pawn"),
        ("NEW_20250805_135338_006_g2.png", "move", "white_knight", "FEN says white_knight, you see white_knight"),
        ("NEW_20250805_135338_007_b2.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_008_e5.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_009_c1.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_009_g2.png", "move", "white_knight", "FEN says white_knight, you see white_knight"),
        ("NEW_20250805_135338_012_e6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_013_e6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_013_g2.png", "move", "white_king", "FEN says white_king, you see white_king"),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/white_bishop"
    
    removed_count = 0
    moved_count = 0
    
    for filename, action, target_dir, reason in fixes:
        source_path = os.path.join(source_dir, filename)
        
        if not os.path.exists(source_path):
            print(f"⚠️  {filename}: File not found")
            continue
        
        if action == "move":
            target_path = f"grey_background_dataset/pieces/test/{target_dir}/{filename}"
            target_dir_path = os.path.dirname(target_path)
            
            # Create target directory if it doesn't exist
            os.makedirs(target_dir_path, exist_ok=True)
            
            # Move the file
            shutil.move(source_path, target_path)
            print(f"📁 {filename}: Moved to {target_dir}/ ({reason})")
            moved_count += 1
            
        elif action == "remove":
            os.remove(source_path)
            print(f"🗑️  {filename}: Removed ({reason})")
            removed_count += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   🗑️  Removed: {removed_count} pieces")
    print(f"   📁 Moved: {moved_count} pieces")
    
    print(f"\n🎉 WHITE BISHOP DIRECTORY CLEANUP COMPLETE!")

if __name__ == "__main__":
    main()
