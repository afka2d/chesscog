#!/usr/bin/env python3
"""
Merge the newly processed enhanced dataset with the existing training dataset.
This ensures all pieces are properly organized for retraining.
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_datasets(enhanced_dir: str, existing_dir: str):
    """
    Merge the enhanced dataset with the existing training dataset.
    
    Args:
        enhanced_dir: Directory containing newly processed pieces
        existing_dir: Directory containing existing training data
    """
    
    # Check if enhanced dataset exists
    if not os.path.exists(enhanced_dir):
        print(f"❌ Enhanced dataset directory not found: {enhanced_dir}")
        return
    
    enhanced_pieces_dir = os.path.join(enhanced_dir, "pieces", "train")
    if not os.path.exists(enhanced_pieces_dir):
        print(f"❌ Enhanced pieces directory not found: {enhanced_pieces_dir}")
        return
    
    # Get piece type folders
    piece_types = [
        'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king',
        'white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king'
    ]
    
    total_pieces_added = 0
    
    print("🔄 Merging enhanced dataset with existing training data...")
    print("=" * 60)
    
    for piece_type in piece_types:
        enhanced_folder = os.path.join(enhanced_pieces_dir, piece_type)
        existing_folder = os.path.join(existing_dir, piece_type)
        
        if os.path.exists(enhanced_folder):
            # Create existing folder if it doesn't exist
            os.makedirs(existing_folder, exist_ok=True)
            
            # Count pieces in enhanced folder
            enhanced_pieces = [f for f in os.listdir(enhanced_folder) if f.endswith('.png')]
            
            if enhanced_pieces:
                print(f"📁 Processing {piece_type}: {len(enhanced_pieces)} pieces")
                
                # Copy each piece to existing folder
                for piece_file in enhanced_pieces:
                    src = os.path.join(enhanced_folder, piece_file)
                    dst = os.path.join(existing_folder, piece_file)
                    
                    # Check if file already exists (avoid duplicates)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        total_pieces_added += 1
                    else:
                        print(f"   ⚠️  Skipping duplicate: {piece_file}")
                
                print(f"   ✅ Added {len(enhanced_pieces)} pieces to {piece_type}")
            else:
                print(f"📁 {piece_type}: No pieces found")
        else:
            print(f"📁 {piece_type}: No enhanced data found")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 MERGE COMPLETE!")
    print(f"♟️  Total pieces added: {total_pieces_added}")
    print(f"📁 Enhanced dataset: {enhanced_dir}")
    print(f"📁 Existing dataset: {existing_dir}")
    
    # Count total pieces in existing dataset
    total_existing = 0
    for piece_type in piece_types:
        existing_folder = os.path.join(existing_dir, piece_type)
        if os.path.exists(existing_folder):
            pieces = [f for f in os.listdir(existing_folder) if f.endswith('.png')]
            total_existing += len(pieces)
            print(f"   {piece_type}: {len(pieces)} pieces")
    
    print(f"\n📊 Total pieces in enhanced dataset: {total_existing}")
    print("💡 Your dataset is now ready for retraining!")

def main():
    """Main function."""
    enhanced_dir = "enhanced_training_dataset"
    existing_dir = "grey_background_dataset/pieces/train"
    
    print("🔄 Chess Dataset Merger")
    print("=" * 30)
    print(f"📁 Enhanced dataset: {enhanced_dir}")
    print(f"📁 Existing dataset: {existing_dir}")
    print("\nThis script will merge newly processed pieces with your existing training data.")
    print("⚠️  IMPORTANT: Your occupancy classifier will remain completely untouched!")
    
    if not os.path.exists(enhanced_dir):
        print(f"\n❌ Enhanced dataset not found: {enhanced_dir}")
        print("Please run the enhanced processor script first.")
        return
    
    if not os.path.exists(existing_dir):
        print(f"\n❌ Existing dataset not found: {existing_dir}")
        print("Please ensure your existing training data is in place.")
        return
    
    input("\nPress Enter to continue with merge...")
    
    merge_datasets(enhanced_dir, existing_dir)

if __name__ == "__main__":
    main()
