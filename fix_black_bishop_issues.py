#!/usr/bin/env python3
"""
Fix misplaced pieces in the black_bishop directory.
This script will move pieces to their correct directories or remove them if they're empty squares.
"""

import os
import shutil

def main():
    """Fix the misplaced pieces in black_bishop directory."""
    print("🔧 FIXING BLACK BISHOP DIRECTORY ISSUES")
    print("=" * 50)
    print("Moving pieces to correct directories or removing empty squares.")
    print()
    
    # Define the fixes needed based on FEN analysis
    fixes = [
        # (source_file, action, target_directory, reason)
        ("NEW_20250805_135338_000_e5.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_000_g4.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_e5.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_g4.png", "move", "black_pawn", "FEN says black_pawn, you see black_pawn"),
        ("NEW_20250805_135338_002_d6.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_002_d7.png", "remove", None, "FEN says empty, you see blank"),
        ("NEW_20250805_135338_003_e6.png", "fix_fen", None, "FEN says black_bishop, you see blank - needs FEN correction"),
        ("NEW_20250805_135338_003_f6.png", "fix_fen", None, "FEN says black_bishop, you see blank - needs FEN correction"),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/black_bishop"
    
    removed_count = 0
    moved_count = 0
    fen_fix_needed = []
    
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
            
        elif action == "fix_fen":
            fen_fix_needed.append(filename)
            print(f"⚠️  {filename}: Needs FEN correction ({reason})")
    
    print(f"\n📊 SUMMARY:")
    print(f"   🗑️  Removed: {removed_count} pieces")
    print(f"   📁 Moved: {moved_count} pieces")
    print(f"   ⚠️  Need FEN correction: {len(fen_fix_needed)} pieces")
    
    if fen_fix_needed:
        print(f"\n⚠️  PIECES NEEDING FEN CORRECTION:")
        for piece in fen_fix_needed:
            print(f"   - {piece}")
        print(f"\n💡 These pieces need their FENs corrected because:")
        print(f"   - The FEN says they should contain black bishops")
        print(f"   - But you see blank squares")
        print(f"   - This suggests the FEN is incorrect for these positions")
    
    print(f"\n🎉 BLACK BISHOP DIRECTORY CLEANUP COMPLETE!")

if __name__ == "__main__":
    main()
