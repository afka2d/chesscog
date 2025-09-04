#!/usr/bin/env python3
"""
Fix misplaced pieces in the black_pawn directory.
Remove empty squares, remove backup files, and keep correctly placed pieces.
"""

import os

def main():
    """Fix misplaced pieces in black_pawn directory."""
    print("🔧 FIXING BLACK PAWN DIRECTORY MISPLACED PIECES")
    print("=" * 60)
    print("Based on FEN analysis, fixing misplaced pieces...")
    print()
    
    # Define the fixes needed based on FEN analysis
    fixes = [
        # (source_file, action, reason)
        ("NEW_20250805_135338_000_b5.png", "remove", "FEN says empty, you see blank"),
        ("NEW_20250805_135338_000_a5.png.backup_fen_fix", "remove", "Backup file - wrong type"),
        ("NEW_20250805_135338_001_b6.png.backup_fen_fix", "remove", "Backup file - wrong type"),
        ("NEW_20250805_135338_001_b7.png", "remove", "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_c7.png", "remove", "FEN says empty, you see blank"),
        ("NEW_20250805_135338_001_f5.png.backup_fen_fix", "remove", "Backup file - wrong type"),
        ("NEW_20250805_135338_001_f7.png", "remove", "FEN says empty, you see blank"),
    ]
    
    source_dir = "grey_background_dataset/pieces/test/black_pawn"
    
    removed_count = 0
    
    for filename, action, reason in fixes:
        source_path = os.path.join(source_dir, filename)
        
        if not os.path.exists(source_path):
            print(f"⚠️  {filename}: File not found")
            continue
        
        if action == "remove":
            os.remove(source_path)
            print(f"🗑️  {filename}: Removed ({reason})")
            removed_count += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   🗑️  Removed: {removed_count} pieces")
    
    print(f"\n🎉 BLACK PAWN DIRECTORY CLEANUP COMPLETE!")

if __name__ == "__main__":
    main()
