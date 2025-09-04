#!/usr/bin/env python3
"""
Fix misplaced pieces in the black_knight directory.
Remove empty squares, move pieces to correct directories, and remove backup files.
"""

import os
import shutil

def main():
    """Fix misplaced pieces in black_knight directory."""
    print("🔧 FIXING BLACK KNIGHT DIRECTORY MISPLACED PIECES")
    print("=" * 60)
    print("Based on FEN analysis, fixing misplaced pieces...")
    print()
    
    # Define the fixes needed based on FEN analysis
    fixes = [
        # (source_file, action, target_directory, reason)
        ("NEW_20250805_135338_001_e4.png.backup_fen_fix", "remove", None, "Backup file - wrong type"),
        ("NEW_20250805_135338_001_f3.png", "move", "white_bishop", "FEN says white_bishop, you see white_bishop"),
        ("NEW_20250805_135338_001_f6.png.backup_fen_fix", "remove", None, "Backup file - wrong type"),
        ("NEW_20250805_135338_002_e4.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_002_f3.png", "move", "white_pawn", "FEN says white_pawn, you see white_pawn"),
        ("NEW_20250805_135338_003_b4.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_003_e5.png", "move", "black_bishop", "FEN says black_bishop, you see black_bishop"),
        ("NEW_20250805_135338_004_d2.png", "move", "white_queen", "FEN says white_queen, you see white_queen"),
        ("NEW_20250805_135338_004_d4.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_005_b6.png", "remove", None, "FEN says empty, you see blank"),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/black_knight"
    
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
    
    print(f"\n🎉 BLACK KNIGHT DIRECTORY CLEANUP COMPLETE!")

if __name__ == "__main__":
    main()
