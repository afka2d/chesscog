#!/usr/bin/env python3
"""
Display sample Marshall piece photos for manual verification
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

def display_sample_pieces(sample_dir="marshall_sample_pieces", num_samples=16):
    """Display a grid of sample piece photos for manual verification"""
    
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        print(f"Sample directory {sample_dir} not found")
        return
    
    # Get all piece images
    piece_files = list(sample_path.glob("*.jpg"))
    if not piece_files:
        print("No piece images found")
        return
    
    # Randomly sample pieces
    sample_files = random.sample(piece_files, min(num_samples, len(piece_files)))
    
    # Create grid
    grid_size = int(num_samples ** 0.5)
    if grid_size * grid_size < num_samples:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i, piece_file in enumerate(sample_files):
        if i >= len(axes):
            break
            
        # Load image
        image = cv2.imread(str(piece_file))
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[i].imshow(image_rgb)
            axes[i].set_title(piece_file.stem, fontsize=8)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'Error loading image', ha='center', va='center')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('marshall_sample_pieces_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Displayed {len(sample_files)} sample pieces")
    print("Check 'marshall_sample_pieces_grid.png' for the complete grid")

def display_pieces_by_type(sample_dir="marshall_sample_pieces"):
    """Display pieces grouped by type for easier verification"""
    
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        print(f"Sample directory {sample_dir} not found")
        return
    
    # Group pieces by type
    piece_groups = {
        'white_rook': [],
        'white_knight': [],
        'white_bishop': [],
        'white_queen': [],
        'white_king': [],
        'white_pawn': [],
        'black_rook': [],
        'black_knight': [],
        'black_bishop': [],
        'black_queen': [],
        'black_king': [],
        'black_pawn': [],
        'empty': []
    }
    
    for piece_file in sample_path.glob("*.jpg"):
        filename = piece_file.stem
        if 'white_R' in filename:
            piece_groups['white_rook'].append(piece_file)
        elif 'white_N' in filename:
            piece_groups['white_knight'].append(piece_file)
        elif 'white_B' in filename:
            piece_groups['white_bishop'].append(piece_file)
        elif 'white_Q' in filename:
            piece_groups['white_queen'].append(piece_file)
        elif 'white_K' in filename:
            piece_groups['white_king'].append(piece_file)
        elif 'white_P' in filename:
            piece_groups['white_pawn'].append(piece_file)
        elif 'black_r' in filename:
            piece_groups['black_rook'].append(piece_file)
        elif 'black_n' in filename:
            piece_groups['black_knight'].append(piece_file)
        elif 'black_b' in filename:
            piece_groups['black_bishop'].append(piece_file)
        elif 'black_q' in filename:
            piece_groups['black_queen'].append(piece_file)
        elif 'black_k' in filename:
            piece_groups['black_king'].append(piece_file)
        elif 'black_p' in filename:
            piece_groups['black_pawn'].append(piece_file)
        elif 'empty' in filename:
            piece_groups['empty'].append(piece_file)
    
    # Display each group
    for piece_type, files in piece_groups.items():
        if not files:
            continue
            
        print(f"\n{piece_type.upper()} ({len(files)} samples):")
        
        # Show first few samples
        for i, piece_file in enumerate(files[:3]):
            print(f"  {i+1}. {piece_file.name}")
        
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")

if __name__ == "__main__":
    print("🔍 Marshall Piece Sample Display")
    print("=" * 40)
    
    # Display random samples
    print("\n📊 Random Sample Grid:")
    display_sample_pieces()
    
    # Display by type
    print("\n📋 Pieces by Type:")
    display_pieces_by_type()
    
    print("\n✅ Sample display complete!")
    print("You can manually verify the accuracy of the piece extraction by examining these images.")
