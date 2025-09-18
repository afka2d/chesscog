#!/usr/bin/env python3
"""
Analyze existing corner annotation data to prepare for corner detection model training.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

class CornerDataAnalyzer:
    def __init__(self):
        self.corner_data = []
        self.image_stats = defaultdict(list)
        
    def analyze_existing_corners(self):
        """Analyze all existing corner annotations"""
        print("🔍 ANALYZING EXISTING CORNER DATA")
        print("=" * 50)
        
        # Find all annotation files
        annotation_dirs = [
            "grey_background_dataset/annotations/train",
            "grey_background_dataset/annotations/val", 
            "grey_background_dataset/annotations/test"
        ]
        
        total_annotations = 0
        
        for ann_dir in annotation_dirs:
            ann_path = Path(ann_dir)
            if ann_path.exists():
                count = self.process_annotation_directory(ann_path)
                total_annotations += count
                print(f"📁 {ann_dir}: {count} annotations")
        
        print(f"\n📊 Total annotations found: {total_annotations}")
        
        if total_annotations > 0:
            self.analyze_corner_patterns()
            self.create_training_plan()
        else:
            print("❌ No corner annotations found")
    
    def process_annotation_directory(self, ann_dir):
        """Process all annotations in a directory"""
        count = 0
        
        for json_file in ann_dir.glob("*.json"):
            # Skip backup files
            if 'backup' in json_file.name:
                continue
                
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                image_name = annotation.get('image', json_file.stem + '.JPG')
                
                if corners and len(corners) == 4:
                    # Find corresponding image
                    image_path = self.find_image_path(image_name, ann_dir)
                    
                    if image_path and image_path.exists():
                        # Load image to get dimensions
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            
                            corner_data = {
                                'image_path': str(image_path),
                                'image_name': image_name,
                                'corners': corners,
                                'image_width': w,
                                'image_height': h,
                                'split': self.get_split_from_path(ann_dir)
                            }
                            
                            self.corner_data.append(corner_data)
                            self.analyze_corner_properties(corner_data)
                            count += 1
                            
            except Exception as e:
                print(f"⚠️  Error processing {json_file}: {e}")
        
        return count
    
    def find_image_path(self, image_name, ann_dir):
        """Find the corresponding image file"""
        # Determine split and construct image path
        if 'train' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/train")
        elif 'val' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/val")
        elif 'test' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/test")
        else:
            return None
        
        image_path = image_dir / image_name
        return image_path if image_path.exists() else None
    
    def get_split_from_path(self, ann_dir):
        """Get the split (train/val/test) from annotation directory path"""
        path_str = str(ann_dir)
        if 'train' in path_str:
            return 'train'
        elif 'val' in path_str:
            return 'val'
        elif 'test' in path_str:
            return 'test'
        return 'unknown'
    
    def analyze_corner_properties(self, corner_data):
        """Analyze properties of corner annotations"""
        corners = corner_data['corners']
        w, h = corner_data['image_width'], corner_data['image_height']
        
        # Convert to numpy array for easier calculation
        corners_np = np.array(corners)
        
        # Calculate board dimensions
        top_left, top_right, bottom_right, bottom_left = corners_np
        
        # Calculate board width and height
        board_width = np.linalg.norm(top_right - top_left)
        board_height = np.linalg.norm(bottom_left - top_left)
        
        # Calculate center
        center = np.mean(corners_np, axis=0)
        
        # Calculate relative positions (normalized by image size)
        rel_corners = corners_np / np.array([w, h])
        
        # Store statistics
        self.image_stats['board_width'].append(board_width)
        self.image_stats['board_height'].append(board_height)
        self.image_stats['board_aspect_ratio'].append(board_width / board_height)
        self.image_stats['center_x'].append(center[0] / w)
        self.image_stats['center_y'].append(center[1] / h)
        self.image_stats['image_aspect_ratio'].append(w / h)
        
        # Store relative corner positions
        for i, corner in enumerate(rel_corners):
            self.image_stats[f'corner_{i}_x'].append(corner[0])
            self.image_stats[f'corner_{i}_y'].append(corner[1])
    
    def analyze_corner_patterns(self):
        """Analyze patterns in corner data"""
        print("\n📊 CORNER PATTERN ANALYSIS")
        print("-" * 30)
        
        # Calculate statistics
        stats = {}
        for key, values in self.image_stats.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Print key statistics
        print(f"Board aspect ratio: {stats['board_aspect_ratio']['mean']:.3f} ± {stats['board_aspect_ratio']['std']:.3f}")
        print(f"Image aspect ratio: {stats['image_aspect_ratio']['mean']:.3f} ± {stats['image_aspect_ratio']['std']:.3f}")
        print(f"Board center X: {stats['center_x']['mean']:.3f} ± {stats['center_x']['std']:.3f}")
        print(f"Board center Y: {stats['center_y']['mean']:.3f} ± {stats['center_y']['std']:.3f}")
        
        # Corner position analysis
        print(f"\nCorner positions (relative to image):")
        for i in range(4):
            x_key = f'corner_{i}_x'
            y_key = f'corner_{i}_y'
            if x_key in stats and y_key in stats:
                print(f"  Corner {i}: ({stats[x_key]['mean']:.3f}, {stats[y_key]['mean']:.3f}) ± ({stats[x_key]['std']:.3f}, {stats[y_key]['std']:.3f})")
        
        # Save detailed statistics
        self.save_corner_statistics(stats)
    
    def create_training_plan(self):
        """Create training plan for corner detection model"""
        print(f"\n🎯 CORNER DETECTION TRAINING PLAN")
        print("-" * 30)
        
        # Count by split
        split_counts = defaultdict(int)
        for data in self.corner_data:
            split_counts[data['split']] += 1
        
        print(f"Training data available:")
        for split, count in split_counts.items():
            print(f"  {split}: {count} images")
        
        total = sum(split_counts.values())
        print(f"  Total: {total} images")
        
        # Training recommendations
        print(f"\n💡 TRAINING RECOMMENDATIONS:")
        
        if total >= 100:
            print("✅ Sufficient data for deep learning model")
            print("   - Recommended: CNN-based corner detection")
            print("   - Architecture: ResNet or EfficientNet backbone")
            print("   - Output: 8 coordinates (4 corners × 2 coordinates)")
        elif total >= 50:
            print("⚠️  Moderate data - consider data augmentation")
            print("   - Recommended: Lightweight CNN + data augmentation")
            print("   - Data augmentation: rotation, scaling, brightness")
        else:
            print("❌ Limited data - consider classical computer vision")
            print("   - Alternative: Improved Hough transform")
            print("   - Alternative: Template matching")
        
        # Model architecture suggestions
        print(f"\n🏗️  SUGGESTED MODEL ARCHITECTURE:")
        print("   Input: RGB image (resized to 512x512)")
        print("   Backbone: EfficientNet-B0 or ResNet18")
        print("   Head: Fully connected layers")
        print("   Output: 8 values [x1,y1,x2,y2,x3,y3,x4,y4]")
        print("   Loss: MSE or Smooth L1 loss")
        
        # Implementation plan
        print(f"\n🚀 IMPLEMENTATION PLAN:")
        print("   1. Create corner detection dataset")
        print("   2. Train corner detection model")
        print("   3. Create new API endpoint (separate from main API)")
        print("   4. Add visualization functionality")
        print("   5. Test without affecting main API")
        
        # Save training data info
        self.save_training_data_info(split_counts, total)
    
    def save_corner_statistics(self, stats):
        """Save corner statistics to file"""
        with open("corner_analysis_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n💾 Corner statistics saved to: corner_analysis_stats.json")
    
    def save_training_data_info(self, split_counts, total):
        """Save training data information"""
        training_info = {
            'total_images': total,
            'split_counts': dict(split_counts),
            'corner_data_sample': self.corner_data[:5],  # First 5 samples
            'recommended_approach': 'CNN-based' if total >= 100 else 'Classical CV' if total < 50 else 'Lightweight CNN'
        }
        
        with open("corner_training_info.json", "w") as f:
            json.dump(training_info, f, indent=2, default=str)
        
        print(f"💾 Training info saved to: corner_training_info.json")
    
    def visualize_corner_distribution(self):
        """Create visualizations of corner distributions"""
        if not self.corner_data:
            print("❌ No corner data to visualize")
            return
        
        print(f"\n📊 Creating corner distribution visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Corner Detection Training Data Analysis')
        
        # Plot 1: Board aspect ratios
        axes[0,0].hist(self.image_stats['board_aspect_ratio'], bins=20, alpha=0.7)
        axes[0,0].set_title('Board Aspect Ratios')
        axes[0,0].set_xlabel('Aspect Ratio')
        axes[0,0].set_ylabel('Frequency')
        
        # Plot 2: Board centers
        axes[0,1].scatter(self.image_stats['center_x'], self.image_stats['center_y'], alpha=0.6)
        axes[0,1].set_title('Board Centers (Relative to Image)')
        axes[0,1].set_xlabel('Center X (relative)')
        axes[0,1].set_ylabel('Center Y (relative)')
        
        # Plot 3: Corner positions
        colors = ['red', 'green', 'blue', 'orange']
        corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        
        for i in range(4):
            x_key = f'corner_{i}_x'
            y_key = f'corner_{i}_y'
            if x_key in self.image_stats and y_key in self.image_stats:
                axes[1,0].scatter(self.image_stats[x_key], self.image_stats[y_key], 
                                c=colors[i], label=corner_names[i], alpha=0.6)
        
        axes[1,0].set_title('Corner Positions (Relative to Image)')
        axes[1,0].set_xlabel('X (relative)')
        axes[1,0].set_ylabel('Y (relative)')
        axes[1,0].legend()
        
        # Plot 4: Image dimensions
        widths = [data['image_width'] for data in self.corner_data]
        heights = [data['image_height'] for data in self.corner_data]
        axes[1,1].scatter(widths, heights, alpha=0.6)
        axes[1,1].set_title('Image Dimensions')
        axes[1,1].set_xlabel('Width (pixels)')
        axes[1,1].set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        plt.savefig('corner_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: corner_analysis.png")

def main():
    """Main function"""
    print("Corner Data Analysis for Training")
    print("=" * 50)
    print("This analyzes your existing corner annotations to prepare")
    print("for training a corner detection model.")
    print()
    
    analyzer = CornerDataAnalyzer()
    analyzer.analyze_existing_corners()
    
    # Optional: Create visualizations
    create_viz = input("\nCreate visualizations? (y/n, default: y): ").strip().lower()
    if create_viz != 'n':
        try:
            analyzer.visualize_corner_distribution()
        except ImportError:
            print("⚠️  Matplotlib not available for visualizations")
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {e}")
    
    print("\n🎯 ANALYSIS COMPLETE!")
    print("Next step: Run corner detection model training")

if __name__ == "__main__":
    main()
