#!/usr/bin/env python3
"""
Show the detailed plan for batch fixing all problematic images.
"""

import os
import json

def analyze_problematic_images():
    """Analyze each problematic image to show what needs fixing."""
    print("🔍 BATCH FIX PLAN - Problematic Images Analysis")
    print("=" * 70)
    
    # Define problematic images from the scan results
    problematic_images = [
        ("NEW_20250805_135337_020", "test"),
        ("NEW_20250805_135337_025", "test"),
        ("IMG_4752", "train"),  # Already fixed in test, but needs train fix
        ("NEW_20250805_135337_023", "train"),
        ("NEW_20250805_135337_042", "train"),
        ("NEW_20250805_135337_013", "train"),
        ("IMG_4754", "train"),  # Also appears in val
        ("IMG_4754", "val"),    # Also appears in train
    ]
    
    # Remove duplicates and prioritize
    unique_images = {}
    for image_name, dataset_type in problematic_images:
        if image_name not in unique_images:
            unique_images[image_name] = dataset_type
        else:
            # Prefer test over train/val, val over train
            if dataset_type == "test":
                unique_images[image_name] = dataset_type
            elif dataset_type == "val" and unique_images[image_name] == "train":
                unique_images[image_name] = dataset_type
    
    print(f"📊 Total problematic images to fix: {len(unique_images)}")
    print(f"🎯 Priority order (TEST → VAL → TRAIN):")
    
    # Group by dataset type
    test_images = []
    val_images = []
    train_images = []
    
    for image_name, dataset_type in unique_images.items():
        if dataset_type == "test":
            test_images.append(image_name)
        elif dataset_type == "val":
            val_images.append(image_name)
        else:
            train_images.append(image_name)
    
    # Show priority order
    priority_order = test_images + val_images + train_images
    
    for i, image_name in enumerate(priority_order, 1):
        dataset_type = unique_images[image_name]
        print(f"   {i:2d}. {image_name} ({dataset_type.upper()})")
    
    print(f"\n📋 DETAILED ANALYSIS:")
    print("=" * 70)
    
    for i, image_name in enumerate(priority_order, 1):
        dataset_type = unique_images[image_name]
        annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
        
        print(f"\n🔍 {i:2d}. {image_name} ({dataset_type.upper()})")
        print(f"   📁 Annotation: {annotation_path}")
        
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                fen = annotation.get('fen', 'N/A')
                
                print(f"   📊 Current corners: {corners}")
                print(f"   📝 Current FEN: {fen}")
                
                # Analyze corner quality
                if len(corners) == 4:
                    corners_np = [corners[0], corners[1], corners[2], corners[3]]
                    board_width = max(c[0] for c in corners_np) - min(c[0] for c in corners_np)
                    board_height = max(c[1] for c in corners_np) - min(c[1] for c in corners_np)
                    aspect_ratio = board_width / board_height if board_height > 0 else 0
                    
                    print(f"   📏 Board dimensions: {board_width:.0f} x {board_height:.0f}")
                    print(f"   🔲 Aspect ratio: {aspect_ratio:.3f}")
                    
                    if 0.95 <= aspect_ratio <= 1.05:
                        print(f"   ✅ Aspect ratio is good")
                    elif 0.9 <= aspect_ratio <= 1.1:
                        print(f"   ⚠️  Aspect ratio is acceptable")
                    else:
                        print(f"   ❌ Aspect ratio needs fixing")
                else:
                    print(f"   ❌ Invalid corner format")
                
            except Exception as e:
                print(f"   ❌ Error reading annotation: {e}")
        else:
            print(f"   ❌ Annotation file not found")
        
        # Check if image exists
        image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
        if os.path.exists(image_path):
            print(f"   📸 Image: ✅ Found")
        else:
            print(f"   📸 Image: ❌ Not found")
    
    print(f"\n🚀 BATCH FIXING PROCESS:")
    print("=" * 70)
    print(f"1. Run: python3 batch_fix_problematic_images.py")
    print(f"2. Script will process images in priority order")
    print(f"3. For each image:")
    print(f"   - Show image for manual corner selection")
    print(f"   - Allow FEN verification/correction")
    print(f"   - Generate corrected piece images")
    print(f"   - Replace dataset pieces")
    print(f"   - Update annotation file")
    print(f"4. Progress tracking and verification")
    print(f"5. Final summary of improvements")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Start with TEST set (highest impact)")
    print(f"   • Take breaks between images if needed")
    print(f"   • Verify each fix before proceeding")
    print(f"   • Keep backups (automatically created)")
    
    print(f"\n🎯 EXPECTED OUTCOME:")
    print(f"   • All 8 problematic images fixed")
    print(f"   • Dataset quality improved from 96.5% to 100%")
    print(f"   • Piece classifier accuracy significantly improved")
    print(f"   • Occupancy classifier remains unchanged")

def main():
    """Main function to show the batch fix plan."""
    try:
        analyze_problematic_images()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
