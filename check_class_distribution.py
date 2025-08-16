#!/usr/bin/env python3
"""
Check class distribution in the training data.
"""

from pathlib import Path
from collections import Counter

def count_pieces():
    """Count pieces in each dataset split."""
    dataset_dir = Path("grey_background_dataset/pieces")
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()} SET:")
        print("-" * 50)
        
        total = 0
        counts = Counter()
        
        split_dir = dataset_dir / split
        for piece_dir in split_dir.glob("*"):
            if piece_dir.is_dir():
                count = len(list(piece_dir.glob("*.png")))
                counts[piece_dir.name] = count
                total += count
        
        # Print counts and percentages
        for piece, count in sorted(counts.items()):
            percentage = (count / total) * 100
            print(f"{piece:15s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\nTotal: {total}")

if __name__ == "__main__":
    count_pieces()