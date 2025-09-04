#!/usr/bin/env python3
"""
Fix black rook issues by identifying source images and providing systematic fixing process.
"""

import os
import json
import shutil
from pathlib import Path

def analyze_problematic_pieces():
    """Analyze the problematic black rook pieces to identify source images."""
    print("🔍 Analyzing problematic black rook pieces...")
    
    # List of problematic pieces provided by user
    problematic_pieces = [
        "IMG_4759_h8.png",
        "IMG_4762_a8.png", 
        "IMG_4769_f8.png",
        "IMG_4775_a8.png",
        "IMG_4784_a8.png",
        "IMG_4788_a2.png",
        "IMG_4822_d8.png",
        "IMG_20250805_135337_002_a8.png",
        "NEW_20250805_135337_011_f8.png",
        "NEW_20250805_135337_020_d8.png",
        "NEW_20250805_135337_025_a8.png",
        "NEW_20250805_135337_030_d8.png",
        "NEW_20250805_135337_033_f8.png",
        "NEW_20250805_135337_037_a8.png",
        "NEW_20250805_135337_063_d8.png",
        "NEW_20250805_135337_071_f8.png",
        "NEW_20250805_135338_000_a8.png",
        "NEW_20250805_135338_000_h5.png",
        "NEW_20250805_135338_001_e8.png",
        "NEW_202508805_135338_002_d8.png",  # Note: typo in original
        "NEW_20250805_135338_002_f8.png",
        "NEW_20250805_135338_004_h8.png",
        "NEW_20250805_135338_005_a8.png",
        "NEW_20250805_135338_005_e7.png",
        "NEW_20250805_135338_006_a8.png",
        "NEW_20250805_135338_006_h5_png",  # Note: missing .png extension
        "NEW_20250805_135338_007_d8.png",
        "NEW_20250805_135338_007_f8.png",
        "NEW_20250805_135338_009_a8.png",
        "NEW_20250805_135338_010_f8.png",
        "NEW_20250805_135338_012_a8.png",
        "NEW_20250805_135338_012_h5.png",
        "NEW_20250805_135338_01_h5.png",   # Note: truncated filename
        "NEW_20250805_135338_072_d8.png",
        "NEW_20250805_135338_077_f8.png",
        "NEW_20250805_135338_080_f8.png",
        "NEW_20250805_135338_084_f8.png"
    ]
    
    # Extract source image names
    source_images = set()
    for piece in problematic_pieces:
        # Handle the typo in the filename
        if "NEW_202508805_135338_002" in piece:
            piece = piece.replace("NEW_202508805_135338_002", "NEW_20250805_135338_002")
        
        # Extract source image name (everything before the last underscore)
        last_underscore = piece.rfind('_')
        if last_underscore != -1:
            source_image = piece[:last_underscore]
            source_images.add(source_image)
    
    print(f"   📊 Found {len(problematic_pieces)} problematic pieces")
    print(f"   📊 From {len(source_images)} unique source images")
    
    # Check which dataset each source image belongs to
    source_image_info = {}
    for source_image in source_images:
        # Check test set
        test_image = f"grey_background_dataset/images/test/{source_image}.JPG"
        test_annotation = f"grey_background_dataset/annotations/test/{source_image}.json"
        
        # Check train set  
        train_image = f"grey_background_dataset/images/train/{source_image}.JPG"
        train_annotation = f"grey_background_dataset/annotations/train/{source_image}.json"
        
        # Check val set
        val_image = f"grey_background_dataset/images/val/{source_image}.JPG"
        val_annotation = f"grey_background_dataset/annotations/val/{source_image}.json"
        
        dataset_type = None
        if os.path.exists(test_image) and os.path.exists(test_annotation):
            dataset_type = "test"
        elif os.path.exists(train_image) and os.path.exists(train_annotation):
            dataset_type = "train"
        elif os.path.exists(val_image) and os.path.exists(val_annotation):
            dataset_type = "val"
        
        if dataset_type:
            source_image_info[source_image] = {
                'dataset': dataset_type,
                'image_path': f"grey_background_dataset/images/{dataset_type}/{source_image}.JPG",
                'annotation_path': f"grey_background_dataset/annotations/{dataset_type}/{source_image}.json"
            }
        else:
            print(f"   ⚠️  {source_image}: Image or annotation not found in any dataset")
    
    return source_image_info, problematic_pieces

def create_progress_tracker():
    """Create a progress tracking file."""
    tracker_file = "black_rook_fix_progress.json"
    
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            "completed_images": [],
            "in_progress_images": [],
            "failed_images": [],
            "total_images": 0,
            "started_at": None,
            "last_updated": None
        }
    
    return tracker_file, progress

def update_progress(tracker_file, progress, image_name, status):
    """Update the progress tracker."""
    import datetime
    
    # Remove from other lists
    if image_name in progress["completed_images"]:
        progress["completed_images"].remove(image_name)
    if image_name in progress["in_progress_images"]:
        progress["in_progress_images"].remove(image_name)
    if image_name in progress["failed_images"]:
        progress["failed_images"].remove(image_name)
    
    # Add to appropriate list
    if status == "completed":
        progress["completed_images"].append(image_name)
    elif status == "in_progress":
        progress["in_progress_images"].append(image_name)
    elif status == "failed":
        progress["failed_images"].append(image_name)
    
    progress["last_updated"] = datetime.datetime.now().isoformat()
    
    with open(tracker_file, 'w') as f:
        json.dump(progress, f, indent=2)

def show_progress_summary(progress):
    """Show current progress summary."""
    print(f"\n📊 PROGRESS SUMMARY:")
    print(f"   Total images to fix: {progress['total_images']}")
    print(f"   Completed: {len(progress['completed_images'])}")
    print(f"   In progress: {len(progress['in_progress_images'])}")
    print(f"   Failed: {len(progress['failed_images'])}")
    print(f"   Remaining: {progress['total_images'] - len(progress['completed_images'])}")
    
    if progress['completed_images']:
        print(f"\n✅ Completed images:")
        for img in progress['completed_images']:
            print(f"      - {img}")
    
    if progress['in_progress_images']:
        print(f"\n🔄 In progress images:")
        for img in progress['in_progress_images']:
            print(f"      - {img}")
    
    if progress['failed_images']:
        print(f"\n❌ Failed images:")
        for img in progress['failed_images']:
            print(f"      - {img}")

def main():
    """Main function to set up black rook fixing process."""
    print("🔧 Black Rook Issues Fix Setup")
    print("=" * 50)
    
    try:
        # Step 1: Analyze problematic pieces
        source_image_info, problematic_pieces = analyze_problematic_pieces()
        
        if not source_image_info:
            print("❌ No source images found for problematic pieces")
            return
        
        # Step 2: Create progress tracker
        tracker_file, progress = create_progress_tracker()
        progress['total_images'] = len(source_image_info)
        progress['started_at'] = progress.get('started_at', None)
        
        # Step 3: Show current status
        show_progress_summary(progress)
        
        # Step 4: Show source images that need fixing
        print(f"\n📋 SOURCE IMAGES TO FIX:")
        print("=" * 50)
        
        remaining_images = []
        for image_name, info in source_image_info.items():
            if image_name not in progress['completed_images']:
                remaining_images.append((image_name, info))
                print(f"   - {image_name} ({info['dataset'].upper()})")
        
        if not remaining_images:
            print("   ✅ All images have been fixed!")
            return
        
        # Step 5: Provide instructions
        print(f"\n💡 NEXT STEPS:")
        print("=" * 50)
        print(f"1. For each image, you'll need to:")
        print(f"   - Manually correct corner coordinates")
        print(f"   - Verify/correct the FEN")
        print(f"   - Regenerate piece images")
        print(f"   - Replace dataset pieces")
        print(f"\n2. Progress is tracked in: {tracker_file}")
        print(f"\n3. Use the existing fixing scripts or create new ones for each image")
        
        # Step 6: Show specific instructions for first image
        if remaining_images:
            first_image, first_info = remaining_images[0]
            print(f"\n🚀 START WITH:")
            print(f"   Image: {first_image}")
            print(f"   Dataset: {first_info['dataset'].upper()}")
            print(f"   Path: {first_info['image_path']}")
            print(f"   Annotation: {first_info['annotation_path']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
