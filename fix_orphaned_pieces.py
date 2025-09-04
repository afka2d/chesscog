#!/usr/bin/env python3
"""
Fix orphaned piece images that have no corresponding annotation files.
This ensures that only accurate, properly annotated pieces are used in training.
"""

import os
import json
import shutil
from pathlib import Path

def find_orphaned_pieces():
    """Find piece images that have no corresponding annotation files."""
    print("🔍 Finding orphaned piece images...")
    
    # Dataset structure
    dataset_types = ['test', 'train', 'val']
    
    orphaned_pieces = {}
    
    for dataset_type in dataset_types:
        pieces_dir = f"grey_background_dataset/pieces/{dataset_type}"
        annotations_dir = f"grey_background_dataset/annotations/{dataset_type}"
        
        if not os.path.exists(pieces_dir):
            continue
            
        print(f"\n🔍 Checking {dataset_type.upper()} set...")
        
        # Get all piece images
        piece_files = []
        for piece_type_dir in os.listdir(pieces_dir):
            piece_type_path = os.path.join(pieces_dir, piece_type_dir)
            if os.path.isdir(piece_type_path):
                for piece_file in os.listdir(piece_type_path):
                    if piece_file.endswith('.png'):
                        piece_files.append(piece_file)
        
        # Extract unique image names from piece filenames
        image_names = set()
        for piece_file in piece_files:
            # Extract image name (e.g., IMG_4752_a2.png -> IMG_4752)
            parts = piece_file.split('_')
            if len(parts) >= 2:
                image_name = '_'.join(parts[:-1])  # Everything except the last part (square)
                image_names.add(image_name)
        
        print(f"   📁 Found {len(piece_files)} piece images from {len(image_names)} source images")
        
        # Check which images have annotations
        orphaned_images = []
        for image_name in image_names:
            annotation_file = f"{image_name}.json"
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            if not os.path.exists(annotation_path):
                orphaned_images.append(image_name)
                print(f"   ❌ {image_name}: No annotation file found")
        
        if orphaned_images:
            orphaned_pieces[dataset_type] = {
                'orphaned_images': orphaned_images,
                'total_pieces': len([f for f in piece_files if any(img in f for img in orphaned_images)])
            }
            print(f"   ⚠️  {len(orphaned_images)} images have orphaned pieces")
        else:
            print(f"   ✅ All images have annotations")
    
    return orphaned_pieces

def clean_orphaned_pieces(orphaned_pieces):
    """Clean up orphaned piece images."""
    print(f"\n🧹 Cleaning up orphaned pieces...")
    
    total_removed = 0
    
    for dataset_type, data in orphaned_pieces.items():
        orphaned_images = data['orphaned_images']
        total_pieces = data['total_pieces']
        
        print(f"\n🔧 Cleaning {dataset_type.upper()} set...")
        print(f"   📁 Will remove {total_pieces} pieces from {len(orphaned_images)} orphaned images")
        
        # Create backup directory
        backup_dir = f"orphaned_pieces_backup/{dataset_type}"
        os.makedirs(backup_dir, exist_ok=True)
        
        removed_count = 0
        
        for image_name in orphaned_images:
            # Find all pieces from this image
            pieces_dir = f"grey_background_dataset/pieces/{dataset_type}"
            
            for piece_type_dir in os.listdir(pieces_dir):
                piece_type_path = os.path.join(pieces_dir, piece_type_dir)
                if os.path.isdir(piece_type_path):
                    for piece_file in os.listdir(piece_type_path):
                        if piece_file.startswith(f"{image_name}_"):
                            piece_path = os.path.join(piece_type_path, piece_file)
                            
                            # Create backup
                            backup_path = os.path.join(backup_dir, f"{piece_type_dir}_{piece_file}")
                            shutil.copy2(piece_path, backup_path)
                            
                            # Remove orphaned piece
                            os.remove(piece_path)
                            removed_count += 1
                            print(f"      🗑️  Removed: {piece_file}")
        
        print(f"   ✅ Removed {removed_count} orphaned pieces")
        print(f"   💾 Backup created in: {backup_dir}")
        total_removed += removed_count
    
    return total_removed

def create_missing_annotation(image_name, dataset_type):
    """Create a basic annotation file for an image that's missing one."""
    print(f"\n📝 Creating missing annotation for {image_name}...")
    
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
    
    # Check if image exists
    image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
    if not os.path.exists(image_path):
        print(f"   ❌ Image not found: {image_path}")
        return False
    
    # Create basic annotation structure
    annotation = {
        "image": f"{image_name}.JPG",
        "corners": [
            [0, 0],      # Placeholder corners
            [1000, 0],   # These will need manual correction
            [1000, 1000],
            [0, 1000]
        ],
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "white_turn": True,
        "timestamp": "placeholder_needs_manual_correction"
    }
    
    # Create backup if annotation already exists
    if os.path.exists(annotation_path):
        backup_path = annotation_path + ".backup_before_creation"
        shutil.copy2(annotation_path, backup_path)
        print(f"   💾 Created backup: {backup_path}")
    
    # Write annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"   ✅ Created annotation: {annotation_path}")
    print(f"   ⚠️  WARNING: This annotation has placeholder corners and FEN!")
    print(f"   🔧 You must manually correct this using the fixing scripts")
    
    return True

def main():
    """Main function to fix orphaned pieces."""
    print("🔧 Orphaned Pieces Cleanup and Fix")
    print("=" * 50)
    
    try:
        # Step 1: Find orphaned pieces
        orphaned_pieces = find_orphaned_pieces()
        
        if not orphaned_pieces:
            print(f"\n✅ No orphaned pieces found! All pieces have proper annotations.")
            return
        
        # Step 2: Show summary
        print(f"\n📊 ORPHANED PIECES SUMMARY:")
        print("=" * 50)
        
        total_orphaned_images = 0
        total_orphaned_pieces = 0
        
        for dataset_type, data in orphaned_pieces.items():
            orphaned_images = data['orphaned_images']
            total_pieces = data['total_pieces']
            
            print(f"\n{dataset_type.upper()}:")
            print(f"   Orphaned images: {len(orphaned_images)}")
            print(f"   Orphaned pieces: {total_pieces}")
            
            for img in orphaned_images:
                print(f"      - {img}")
            
            total_orphaned_images += len(orphaned_images)
            total_orphaned_pieces += total_pieces
        
        print(f"\n🎯 TOTAL IMPACT:")
        print(f"   Orphaned images: {total_orphaned_images}")
        print(f"   Orphaned pieces: {total_orphaned_pieces}")
        
        # Step 3: Ask user what to do
        print(f"\n💡 OPTIONS:")
        print(f"   1. Clean up ALL orphaned pieces (recommended)")
        print(f"   2. Create placeholder annotations for missing images")
        print(f"   3. Both cleanup and create placeholders")
        print(f"   4. Exit without changes")
        
        while True:
            try:
                choice = input(f"\n🚀 Choose option (1-4): ").strip()
                
                if choice == '1':
                    # Clean up only
                    removed = clean_orphaned_pieces(orphaned_pieces)
                    print(f"\n✅ Cleanup complete! Removed {removed} orphaned pieces.")
                    break
                    
                elif choice == '2':
                    # Create placeholders only
                    print(f"\n📝 Creating placeholder annotations...")
                    for dataset_type, data in orphaned_pieces.items():
                        for image_name in data['orphaned_images']:
                            create_missing_annotation(image_name, dataset_type)
                    print(f"\n✅ Placeholder annotations created!")
                    break
                    
                elif choice == '3':
                    # Both
                    removed = clean_orphaned_pieces(orphaned_pieces)
                    print(f"\n📝 Creating placeholder annotations...")
                    for dataset_type, data in orphaned_pieces.items():
                        for image_name in data['orphaned_images']:
                            create_missing_annotation(image_name, dataset_type)
                    print(f"\n✅ Complete! Removed {removed} orphaned pieces and created placeholders.")
                    break
                    
                elif choice == '4':
                    print(f"\n❌ No changes made. Exiting.")
                    return
                    
                else:
                    print(f"   ❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print(f"\n❌ Operation cancelled.")
                return
        
        # Step 4: Final recommendations
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Run the dataset scanner to verify cleanup")
        print(f"   2. For any placeholder annotations, use the fixing scripts")
        print(f"   3. Ensure all pieces have accurate corner coordinates and FEN")
        print(f"   4. Only then proceed with training")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
