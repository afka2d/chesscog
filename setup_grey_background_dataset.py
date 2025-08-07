#!/usr/bin/env python3
"""
Setup Grey Background Chess Dataset

This script sets up a fresh training dataset specifically for chess images with grey backgrounds.
It creates the necessary directory structure and provides guidance for adding your images.
"""

import os
import argparse
from pathlib import Path

def create_directory_structure():
    """Create the directory structure for the grey background dataset"""
    
    # Main dataset directory
    dataset_dir = "grey_background_dataset"
    
    # Create directories
    directories = [
        dataset_dir,
        f"{dataset_dir}/images",
        f"{dataset_dir}/annotations",
        f"{dataset_dir}/processed",
        f"{dataset_dir}/models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return dataset_dir

def create_readme(dataset_dir):
    """Create a README file with instructions"""
    
    readme_content = """# Grey Background Chess Dataset

This dataset is specifically designed for training chess recognition models on images with grey backgrounds.

## Directory Structure

```
grey_background_dataset/
├── images/           # Place your grey background chess images here
├── annotations/      # JSON annotation files (auto-generated)
├── processed/        # Processed training data
└── models/          # Trained models
```

## Setup Instructions

1. **Add Images**: Place your grey background chess images in the `images/` directory
   - Supported formats: JPG, JPEG, PNG
   - Images should have grey backgrounds for consistent training

2. **Create Annotations**: Run the annotation creation script:
   ```bash
   python create_custom_dataset.py --input_dir grey_background_dataset/images --output_dir grey_background_dataset
   ```

3. **Update Corner Coordinates**: Use the corner update tool:
   ```bash
   python update_corners.py --interactive
   ```

4. **Train Models**: Run the training script:
   ```bash
   python enhanced_batch_train.py --full_pipeline
   ```

## Image Requirements

- **Background**: Grey background (consistent lighting)
- **Chess Board**: Clear 8x8 grid visible
- **Pieces**: Standard chess pieces clearly visible
- **Quality**: Good resolution, well-lit, no blur

## Expected Results

Training on grey background images should improve recognition accuracy for:
- Images with similar grey backgrounds
- Consistent lighting conditions
- Standard chess board setups

## Notes

- This dataset is optimized for grey background scenarios
- Corner coordinates are crucial for accurate training
- FEN notation should be provided for each position
"""
    
    readme_path = f"{dataset_dir}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_path}")

def create_sample_annotation_template(dataset_dir):
    """Create a sample annotation template"""
    
    template_content = """{
  "image_path": "grey_background_dataset/images/your_image.jpg",
  "image_size": [width, height],
  "corners": [
    [x1, y1],  // Top-left corner of chess board
    [x2, y2],  // Top-right corner of chess board
    [x3, y3],  // Bottom-right corner of chess board
    [x4, y4]   // Bottom-left corner of chess board
  ],
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "white_turn": true,
  "notes": "Sample annotation for grey background chess image",
  "description": "Description of the chess position"
}"""
    
    template_path = f"{dataset_dir}/sample_annotation.json"
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    print(f"✅ Created sample annotation template: {template_path}")

def create_quick_start_script(dataset_dir):
    """Create a quick start script for the dataset"""
    
    script_content = f"""#!/bin/bash
# Quick Start Script for Grey Background Chess Dataset

echo "🎯 Setting up Grey Background Chess Dataset"

# Step 1: Check if images are present
if [ ! "$(ls -A {dataset_dir}/images)" ]; then
    echo "❌ No images found in {dataset_dir}/images/"
    echo "Please add your grey background chess images to {dataset_dir}/images/"
    exit 1
fi

echo "✅ Found images in {dataset_dir}/images/"

# Step 2: Create annotations
echo "📝 Creating annotations..."
python create_custom_dataset.py --input_dir {dataset_dir}/images --output_dir {dataset_dir}

# Step 3: Show next steps
echo ""
echo "🎯 Next Steps:"
echo "1. Update corner coordinates: python update_corners.py --interactive"
echo "2. Add FEN notations to annotation files"
echo "3. Train the model: python enhanced_batch_train.py --full_pipeline"
echo ""
echo "📖 See {dataset_dir}/README.md for detailed instructions"
"""
    
    script_path = f"{dataset_dir}/quick_start.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created quick start script: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Setup Grey Background Chess Dataset")
    parser.add_argument("--clean", action="store_true", help="Clean existing dataset first")
    
    args = parser.parse_args()
    
    print("🎯 Setting up Grey Background Chess Dataset")
    print("=" * 50)
    
    # Clean if requested
    if args.clean:
        import shutil
        if os.path.exists("grey_background_dataset"):
            shutil.rmtree("grey_background_dataset")
            print("🧹 Cleaned existing dataset")
    
    # Create directory structure
    dataset_dir = create_directory_structure()
    
    # Create documentation
    create_readme(dataset_dir)
    create_sample_annotation_template(dataset_dir)
    create_quick_start_script(dataset_dir)
    
    print("\n🎉 Dataset setup complete!")
    print("\n📋 Next Steps:")
    print("1. Add your grey background chess images to grey_background_dataset/images/")
    print("2. Run: ./grey_background_dataset/quick_start.sh")
    print("3. Update corner coordinates using the interactive tool")
    print("4. Train your model")
    
    print(f"\n📁 Dataset location: {dataset_dir}/")
    print(f"📖 Documentation: {dataset_dir}/README.md")

if __name__ == "__main__":
    main() 