#!/usr/bin/env python3
"""
Script to expand the chess piece dataset with synthetic data generation
and data collection tools to achieve 90% accuracy.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import shutil

class ChessPieceGenerator:
    """Generate synthetic chess piece images."""
    
    def __init__(self, output_dir="expanded_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Chess piece symbols (Unicode)
        self.pieces = {
            'white_king': '♔', 'white_queen': '♕', 'white_rook': '♖',
            'white_bishop': '♗', 'white_knight': '♘', 'white_pawn': '♙',
            'black_king': '♚', 'black_queen': '♛', 'black_rook': '♜',
            'black_bishop': '♝', 'black_knight': '♞', 'black_pawn': '♟'
        }
        
        # Colors for pieces
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        # Background colors
        self.backgrounds = [
            (128, 128, 128),  # Gray
            (200, 200, 200),  # Light gray
            (100, 100, 100),  # Dark gray
            (150, 150, 150),  # Medium gray
        ]
    
    def generate_piece_image(self, piece_name, size=(224, 448), font_size=120):
        """Generate a synthetic chess piece image."""
        # Create background
        bg_color = random.choice(self.backgrounds)
        img = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Get piece symbol
        symbol = self.pieces[piece_name]
        
        # Try to load a chess font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate position to center the piece
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw the piece
        color = self.colors['white'] if piece_name.startswith('white') else self.colors['black']
        draw.text((x, y), symbol, fill=color, font=font)
        
        return img
    
    def apply_variations(self, img, piece_name):
        """Apply various transformations to create diverse images."""
        variations = []
        
        # Original image
        variations.append(img.copy())
        
        # Rotation variations
        for angle in [-15, -10, -5, 5, 10, 15]:
            rotated = img.rotate(angle, fillcolor=random.choice(self.backgrounds))
            variations.append(rotated)
        
        # Brightness variations
        for factor in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            enhancer = ImageEnhance.Brightness(img)
            brightened = enhancer.enhance(factor)
            variations.append(brightened)
        
        # Contrast variations
        for factor in [0.8, 0.9, 1.1, 1.2]:
            enhancer = ImageEnhance.Contrast(img)
            contrasted = enhancer.enhance(factor)
            variations.append(contrasted)
        
        # Blur variations (simulate focus issues)
        for radius in [0.5, 1.0, 1.5]:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
            variations.append(blurred)
        
        # Noise variations
        for noise_level in [2, 5, 8]:
            noisy = self.add_noise(img, noise_level)
            variations.append(noisy)
        
        # Perspective variations
        for _ in range(3):
            perspective = self.apply_perspective(img)
            variations.append(perspective)
        
        return variations
    
    def add_noise(self, img, noise_level):
        """Add random noise to image."""
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.uint8)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def apply_perspective(self, img):
        """Apply perspective transformation."""
        width, height = img.size
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Define destination points with some perspective distortion
        offset = random.randint(10, 30)
        dst_points = np.float32([
            [offset, offset],
            [width - offset, offset],
            [width - offset, height - offset],
            [offset, height - offset]
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        img_array = np.array(img)
        transformed = cv2.warpPerspective(img_array, matrix, (width, height))
        
        return Image.fromarray(transformed)
    
    def generate_dataset(self, target_samples_per_class=1000):
        """Generate synthetic dataset with target number of samples per class."""
        print(f"Generating synthetic dataset with {target_samples_per_class} samples per class...")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for piece_name in self.pieces.keys():
                piece_dir = self.output_dir / split / piece_name
                piece_dir.mkdir(parents=True, exist_ok=True)
        
        for piece_name in self.pieces.keys():
            print(f"Generating {piece_name}...")
            
            # Generate base images
            base_images = []
            for _ in range(target_samples_per_class // 20):  # Generate fewer base images
                base_img = self.generate_piece_image(piece_name)
                base_images.append(base_img)
            
            # Apply variations to create more samples
            all_variations = []
            for base_img in base_images:
                variations = self.apply_variations(base_img, piece_name)
                all_variations.extend(variations)
            
            # Shuffle and limit to target number
            random.shuffle(all_variations)
            all_variations = all_variations[:target_samples_per_class]
            
            # Split into train/val/test (70/15/15)
            train_count = int(target_samples_per_class * 0.7)
            val_count = int(target_samples_per_class * 0.15)
            
            train_images = all_variations[:train_count]
            val_images = all_variations[train_count:train_count + val_count]
            test_images = all_variations[train_count + val_count:]
            
            # Save images
            for i, img in enumerate(train_images):
                img.save(self.output_dir / 'train' / piece_name / f'{piece_name}_syn_{i:04d}.png')
            
            for i, img in enumerate(val_images):
                img.save(self.output_dir / 'val' / piece_name / f'{piece_name}_syn_{i:04d}.png')
            
            for i, img in enumerate(test_images):
                img.save(self.output_dir / 'test' / piece_name / f'{piece_name}_syn_{i:04d}.png')
            
            print(f"  Generated {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")

class DatasetAnalyzer:
    """Analyze and visualize dataset statistics."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
    
    def count_samples(self):
        """Count samples in each class and split."""
        counts = {}
        
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            counts[split] = {}
            for piece_dir in split_path.iterdir():
                if piece_dir.is_dir():
                    piece_name = piece_dir.name
                    image_count = len(list(piece_dir.glob('*.png'))) + len(list(piece_dir.glob('*.jpg')))
                    counts[split][piece_name] = image_count
        
        return counts
    
    def plot_distribution(self, counts):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, split in enumerate(['train', 'val', 'test']):
            if split not in counts:
                continue
                
            classes = list(counts[split].keys())
            values = list(counts[split].values())
            
            axes[i].bar(classes, values)
            axes[i].set_title(f'{split.capitalize()} Set Distribution')
            axes[i].set_xlabel('Piece Type')
            axes[i].set_ylabel('Number of Samples')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_statistics(self, counts):
        """Print dataset statistics."""
        print("\nDataset Statistics:")
        print("=" * 50)
        
        for split in ['train', 'val', 'test']:
            if split not in counts:
                continue
                
            print(f"\n{split.upper()} SET:")
            total = sum(counts[split].values())
            print(f"Total samples: {total}")
            
            for piece, count in counts[split].items():
                percentage = (count / total) * 100
                print(f"  {piece}: {count} ({percentage:.1f}%)")

class DatasetMerger:
    """Merge original and synthetic datasets."""
    
    def __init__(self, original_path, synthetic_path, output_path):
        self.original_path = Path(original_path)
        self.synthetic_path = Path(synthetic_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    def merge_datasets(self):
        """Merge original and synthetic datasets."""
        print("Merging original and synthetic datasets...")
        
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            
            # Create output directories
            for piece_dir in (self.original_path / split).iterdir():
                if piece_dir.is_dir():
                    output_piece_dir = self.output_path / split / piece_dir.name
                    output_piece_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy original images
            original_count = 0
            for piece_dir in (self.original_path / split).iterdir():
                if piece_dir.is_dir():
                    piece_name = piece_dir.name
                    output_piece_dir = self.output_path / split / piece_name
                    
                    # Copy original images
                    for img_path in piece_dir.glob('*.png'):
                        shutil.copy2(img_path, output_piece_dir / f"orig_{img_path.name}")
                        original_count += 1
                    
                    for img_path in piece_dir.glob('*.jpg'):
                        shutil.copy2(img_path, output_piece_dir / f"orig_{img_path.name}")
                        original_count += 1
            
            # Copy synthetic images
            synthetic_count = 0
            if (self.synthetic_path / split).exists():
                for piece_dir in (self.synthetic_path / split).iterdir():
                    if piece_dir.is_dir():
                        piece_name = piece_dir.name
                        output_piece_dir = self.output_path / split / piece_name
                        
                        # Copy synthetic images
                        for img_path in piece_dir.glob('*.png'):
                            shutil.copy2(img_path, output_piece_dir / f"syn_{img_path.name}")
                            synthetic_count += 1
            
            print(f"  Original images: {original_count}")
            print(f"  Synthetic images: {synthetic_count}")
            print(f"  Total: {original_count + synthetic_count}")

def main():
    """Main function to expand the dataset."""
    print("Chess Dataset Expansion Tool")
    print("=" * 50)
    
    # Configuration
    target_samples_per_class = 1000  # Target 1000 samples per class
    original_dataset = "grey_background_dataset/pieces"
    synthetic_dataset = "expanded_dataset"
    merged_dataset = "enhanced_chess_dataset"
    
    # Step 1: Generate synthetic data
    print("\nStep 1: Generating synthetic chess piece images...")
    generator = ChessPieceGenerator(synthetic_dataset)
    generator.generate_dataset(target_samples_per_class)
    
    # Step 2: Analyze original dataset
    print("\nStep 2: Analyzing original dataset...")
    analyzer = DatasetAnalyzer(original_dataset)
    original_counts = analyzer.count_samples()
    analyzer.print_statistics(original_counts)
    
    # Step 3: Analyze synthetic dataset
    print("\nStep 3: Analyzing synthetic dataset...")
    synthetic_analyzer = DatasetAnalyzer(synthetic_dataset)
    synthetic_counts = synthetic_analyzer.count_samples()
    synthetic_analyzer.print_statistics(synthetic_counts)
    
    # Step 4: Merge datasets
    print("\nStep 4: Merging original and synthetic datasets...")
    merger = DatasetMerger(original_dataset, synthetic_dataset, merged_dataset)
    merger.merge_datasets()
    
    # Step 5: Analyze final dataset
    print("\nStep 5: Analyzing final merged dataset...")
    final_analyzer = DatasetAnalyzer(merged_dataset)
    final_counts = final_analyzer.count_samples()
    final_analyzer.print_statistics(final_counts)
    final_analyzer.plot_distribution(final_counts)
    
    print(f"\nDataset expansion complete!")
    print(f"Original dataset: {original_dataset}")
    print(f"Synthetic dataset: {synthetic_dataset}")
    print(f"Merged dataset: {merged_dataset}")
    print(f"Use the merged dataset for training the enhanced model.")

if __name__ == "__main__":
    main() 