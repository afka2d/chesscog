#!/usr/bin/env python3
"""
Clean up Marshall photos directory by removing problematic files
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import pillow_heif

def test_image_load(image_path):
    """Test if an image can be loaded successfully"""
    try:
        if image_path.suffix.lower() == '.heic':
            pillow_heif.register_heif_opener()
            with Image.open(image_path) as img:
                img.verify()  # Verify the image
                return True
        else:
            with Image.open(image_path) as img:
                img.verify()
                return True
    except Exception as e:
        print(f"❌ Error loading {image_path.name}: {e}")
        return False

def clean_marshall_photos():
    """Clean up the Marshall photos directory"""
    marshall_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    backup_dir = Path("/Users/tonyblum/Desktop/marshall photos - backup")
    error_dir = Path("/Users/tonyblum/Desktop/marshall photos - errors")
    
    if not marshall_dir.exists():
        print(f"❌ Marshall photos directory not found: {marshall_dir}")
        return
    
    # Create backup and error directories
    backup_dir.mkdir(exist_ok=True)
    error_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Scanning {marshall_dir} for problematic images...")
    
    # Find all image files
    image_extensions = ['.heic', '.HEIC', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(marshall_dir.glob(f"*{ext}"))
    
    print(f"📊 Found {len(image_files)} image files")
    
    # Test each image
    good_images = []
    bad_images = []
    
    for i, image_path in enumerate(image_files):
        print(f"🔍 Testing {i+1}/{len(image_files)}: {image_path.name}")
        
        if test_image_load(image_path):
            good_images.append(image_path)
            print(f"✅ {image_path.name} - OK")
        else:
            bad_images.append(image_path)
            print(f"❌ {image_path.name} - ERROR")
    
    print(f"\n📊 RESULTS:")
    print(f"   Good images: {len(good_images)}")
    print(f"   Bad images: {len(bad_images)}")
    
    if bad_images:
        print(f"\n🗑️  Moving {len(bad_images)} problematic images to error directory...")
        for bad_image in bad_images:
            try:
                # Move to error directory
                shutil.move(str(bad_image), str(error_dir / bad_image.name))
                print(f"   Moved {bad_image.name} to error directory")
            except Exception as e:
                print(f"   Failed to move {bad_image.name}: {e}")
    
    # Create a backup of good images
    print(f"\n💾 Creating backup of good images...")
    for good_image in good_images:
        try:
            backup_path = backup_dir / good_image.name
            if not backup_path.exists():
                shutil.copy2(str(good_image), str(backup_path))
        except Exception as e:
            print(f"   Failed to backup {good_image.name}: {e}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   Good images: {len(good_images)} (backed up to {backup_dir})")
    print(f"   Bad images: {len(bad_images)} (moved to {error_dir})")
    print(f"   Original directory now contains only good images")

if __name__ == "__main__":
    clean_marshall_photos()

