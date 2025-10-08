#!/usr/bin/env python3
"""
Convert HEIC images from marshall2 folder to JPG format and set up annotation structure.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(heic_path, jpg_path):
    """Convert HEIC image to JPG"""
    try:
        # Register HEIF opener
        pillow_heif.register_heif_opener()
        
        # Open HEIC image
        image = Image.open(heic_path)
        
        # Convert to RGB (JPG doesn't support transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG
        image.save(jpg_path, 'JPEG', quality=95)
        return True
        
    except Exception as e:
        print(f"❌ Error converting {heic_path}: {e}")
        return False

def setup_marshall2_annotation():
    """Set up annotation structure for marshall2 images"""
    
    # Source and destination directories
    source_dir = Path("/Users/tonyblum/Desktop/marshall2")
    dest_dir = Path("marshall2_training_images")
    
    # Create directory structure
    dest_dir.mkdir(exist_ok=True)
    (dest_dir / "annotations").mkdir(exist_ok=True)
    (dest_dir / "visualizations").mkdir(exist_ok=True)
    
    print(f"🔄 Converting HEIC images from {source_dir}")
    print(f"📁 Destination: {dest_dir}")
    
    # Get all HEIC files
    heic_files = list(source_dir.glob("*.HEIC"))
    print(f"📊 Found {len(heic_files)} HEIC images")
    
    converted_count = 0
    skipped_count = 0
    
    for heic_file in heic_files:
        # Create JPG filename
        jpg_file = dest_dir / f"{heic_file.stem}.jpg"
        
        # Skip if JPG already exists
        if jpg_file.exists():
            print(f"⏭️  Skipping {heic_file.name} (JPG already exists)")
            skipped_count += 1
            continue
        
        # Convert HEIC to JPG
        print(f"🔄 Converting {heic_file.name}...")
        if convert_heic_to_jpg(heic_file, jpg_file):
            print(f"✅ Converted to {jpg_file.name}")
            converted_count += 1
        else:
            print(f"❌ Failed to convert {heic_file.name}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"   ✅ Converted: {converted_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   📁 Total images: {len(list(dest_dir.glob('*.jpg')))}")
    
    return dest_dir

def create_annotation_templates(dest_dir):
    """Create annotation template files for each image"""
    
    annotations_dir = dest_dir / "annotations"
    image_files = list(dest_dir.glob("*.jpg"))
    
    print(f"\n📝 Creating annotation templates...")
    
    for image_file in image_files:
        annotation_file = annotations_dir / f"{image_file.stem}.json"
        
        if annotation_file.exists():
            print(f"⏭️  Skipping {image_file.name} (annotation exists)")
            continue
        
        # Create annotation template
        annotation = {
            "image_path": str(image_file),
            "image_name": image_file.name,
            "chess_set": "marshall2",
            "corners": [
                [0, 0],      # Top-left (to be annotated)
                [1000, 0],   # Top-right (to be annotated)
                [1000, 1000], # Bottom-right (to be annotated)
                [0, 1000]    # Bottom-left (to be annotated)
            ],
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "white_turn": True,
            "notes": "Please annotate the 4 corner points and provide the correct FEN notation",
            "status": "pending"
        }
        
        # Save annotation template
        with open(annotation_file, 'w') as f:
            import json
            json.dump(annotation, f, indent=2)
        
        print(f"📝 Created template: {annotation_file.name}")
    
    print(f"✅ Created {len(list(annotations_dir.glob('*.json')))} annotation templates")

def main():
    """Main function"""
    print("🎯 Marshall2 Image Conversion and Annotation Setup")
    print("=" * 60)
    
    # Step 1: Convert HEIC to JPG
    dest_dir = setup_marshall2_annotation()
    
    # Step 2: Create annotation templates
    create_annotation_templates(dest_dir)
    
    print(f"\n🎉 Setup Complete!")
    print(f"📁 Images ready for annotation: {dest_dir}")
    print(f"📝 Annotation templates: {dest_dir}/annotations/")
    print(f"")
    print(f"🚀 Next steps:")
    print(f"   1. Run annotation tool: python3 visual_chess_annotator.py")
    print(f"   2. Or use simple annotator: python3 simple_annotation_tool.py")
    print(f"   3. Or use semi-automated tool: python3 semi_automated_annotation_tool.py")

if __name__ == "__main__":
    main()
