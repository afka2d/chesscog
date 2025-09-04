#!/usr/bin/env python3
"""
Fix misplaced pieces in the black_rook directory.
This script will move pieces to their correct directories or remove them if they're empty squares.
"""

import os
import shutil

def main():
    """Fix the misplaced pieces identified by the user."""
    print("🔧 FIXING MISPLACED PIECES")
    print("=" * 50)
    print("Moving pieces from black_rook directory to their correct locations.")
    print()
    
    # Define the fixes needed
    fixes = [
        # (source_file, action, target_directory)
        ("NEW_20250805_135338_000_h5.png", "move", "black_pawn"),
        ("NEW_20250805_135338_001_e8.png", "remove", None),
        ("NEW_20250805_135338_002_d8.png", "remove", None),
        ("NEW_20250805_135338_002_f8.png", "remove", None),
        ("NEW_20250805_135338_003_f8.png", "keep", "black_rook"),  # This one is correct
        ("NEW_20250805_135338_004_h8.png", "remove", None),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/black_rook"
    
    for filename, action, target_dir in fixes:
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
            print(f"✅ {filename}: Moved to {target_dir}/")
            
        elif action == "remove":
            os.remove(source_path)
            print(f"🗑️  {filename}: Removed (empty square)")
            
        elif action == "keep":
            print(f"✅ {filename}: Already in correct location")
    
    print(f"\n🎉 MISPLACED PIECES FIXED!")
    print("All pieces are now in their correct directories.")

if __name__ == "__main__":
    main()
