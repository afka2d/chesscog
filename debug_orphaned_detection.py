#!/usr/bin/env python3
"""
Debug script to investigate orphaned piece detection logic.
"""

import os
import json

def debug_filename_parsing():
    """Debug how piece filenames are being parsed."""
    print("🔍 Debugging filename parsing...")
    
    # Test with a few piece filenames
    test_filenames = [
        "IMG_4679_d2.png",
        "NEW_20250805_135337_000_c2.png",
        "NEW_20250805_135337_001_b3.png",
        "IMG_4752_a2.png"
    ]
    
    print(f"\n📋 Testing filename parsing:")
    for filename in test_filenames:
        # My current logic
        parts = filename.split('_')
        if len(parts) >= 2:
            image_name = '_'.join(parts[:-1])  # Everything except the last part (square)
        else:
            image_name = filename
        
        print(f"   {filename}")
        print(f"      Parts: {parts}")
        print(f"      Extracted image name: {image_name}")
        
        # Check if annotation exists
        annotation_path = f"grey_background_dataset/annotations/train/{image_name}.json"
        exists = os.path.exists(annotation_path)
        print(f"      Annotation exists: {exists}")
        print()

def debug_actual_annotations():
    """Check what annotations actually exist vs what pieces exist."""
    print("🔍 Checking actual annotations vs pieces...")
    
    # Get all annotation files
    annotations_dir = "grey_background_dataset/annotations/train"
    annotation_files = [f.replace('.json', '') for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    print(f"   📁 Found {len(annotation_files)} annotation files")
    
    # Get sample piece images
    pieces_dir = "grey_background_dataset/pieces/train/white_pawn"
    piece_files = [f for f in os.listdir(pieces_dir) if f.endswith('.png')]
    
    print(f"   📁 Found {len(piece_files)} white pawn pieces")
    
    # Extract image names from piece files
    piece_image_names = set()
    for piece_file in piece_files:
        # Correct parsing: split by last underscore to separate image name from square
        last_underscore = piece_file.rfind('_')
        if last_underscore != -1:
            image_name = piece_file[:last_underscore]
            square = piece_file[last_underscore + 1:].replace('.png', '')
            piece_image_names.add(image_name)
    
    print(f"   📊 Extracted {len(piece_image_names)} unique image names from pieces")
    
    # Check which piece images have annotations
    missing_annotations = []
    for image_name in piece_image_names:
        if image_name not in annotation_files:
            missing_annotations.append(image_name)
    
    print(f"   ❌ {len(missing_annotations)} piece images missing annotations:")
    for img in missing_annotations[:10]:  # Show first 10
        print(f"      - {img}")
    if len(missing_annotations) > 10:
        print(f"      ... and {len(missing_annotations) - 10} more")
    
    # Check which annotations have no pieces
    missing_pieces = []
    for image_name in annotation_files:
        if image_name not in piece_image_names:
            missing_pieces.append(image_name)
    
    print(f"   ❌ {len(missing_pieces)} annotations missing pieces:")
    for img in missing_pieces[:10]:  # Show first 10
        print(f"      - {img}")
    if len(missing_pieces) > 10:
        print(f"      ... and {len(missing_pieces) - 10} more")

def test_corrected_parsing():
    """Test the corrected filename parsing logic."""
    print(f"\n🔧 Testing corrected parsing logic...")
    
    pieces_dir = "grey_background_dataset/pieces/train/white_pawn"
    piece_files = [f for f in os.listdir(pieces_dir) if f.endswith('.png')]
    
    # Test with corrected logic
    piece_image_names = set()
    for piece_file in piece_files:
        # Correct parsing: split by last underscore
        last_underscore = piece_file.rfind('_')
        if last_underscore != -1:
            image_name = piece_file[:last_underscore]
            piece_image_names.add(image_name)
    
    print(f"   📊 Corrected parsing found {len(piece_image_names)} image names")
    
    # Check annotations
    annotations_dir = "grey_background_dataset/annotations/train"
    annotation_files = [f.replace('.json', '') for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    missing_annotations = []
    for image_name in piece_image_names:
        if image_name not in annotation_files:
            missing_annotations.append(image_name)
    
    print(f"   ❌ {len(missing_annotations)} piece images missing annotations:")
    for img in missing_annotations[:5]:  # Show first 5
        print(f"      - {img}")

def main():
    """Main debug function."""
    print("🔍 Orphaned Piece Detection Debug")
    print("=" * 50)
    
    try:
        # Step 1: Debug filename parsing
        debug_filename_parsing()
        
        # Step 2: Check actual annotations vs pieces
        debug_actual_annotations()
        
        # Step 3: Test corrected parsing
        test_corrected_parsing()
        
        print(f"\n💡 SUMMARY:")
        print(f"   The bug was in the filename parsing logic.")
        print(f"   I was using split('_')[:-1] which doesn't work for")
        print(f"   filenames with multiple underscores like NEW_20250805_135337_000")
        print(f"   The correct approach is to use rfind('_') to find the last underscore.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
