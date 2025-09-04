#!/usr/bin/env python3
"""
Fix misplaced pieces in the black_queen directory.
Remove empty squares, move pieces to correct directories, and remove backup files.
"""

import os
import shutil

def main():
    """Fix misplaced pieces in black_queen directory."""
    print("🔧 FIXING BLACK QUEEN DIRECTORY MISPLACED PIECES")
    print("=" * 60)
    print("Based on FEN analysis, fixing misplaced pieces...")
    print()
    
    # Define the fixes needed based on FEN analysis
    fixes = [
        # (source_file, action, target_directory, reason)
        ("NEW_20250805_135338_000_c5.png", "move", "black_bishop", "FEN says black_bishop, you see black_bishop"),
        ("NEW_20250805_135338_001_f6.png", "move", "black_knight", "FEN says black_knight, you see black_knight"),
        ("NEW_20250805_135338_001_g6.png.backup_fen_fix", "remove", None, "Backup file - wrong type"),
        ("NEW_20250805_135338_003_g6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_004_e3.png", "move", "white_bishop", "FEN says white_bishop, you see white_bishop"),
        ("NEW_20250805_135338_005_g6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_006_e3.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_007_e7.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_010_d8.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_011_e3.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_012_c5.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_013_c5.png", "remove", None, "FEN says empty, you see blank"),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/black_queen"
    
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
    
    print(f"\n🎉 BLACK QUEEN DIRECTORY CLEANUP COMPLETE!")

if __name__ == "__main__":
    main()
