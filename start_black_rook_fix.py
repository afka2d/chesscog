#!/usr/bin/env python3
"""
Start the black rook fixing process with the first image.
"""

import os
import json

def main():
    """Start fixing the first black rook problematic image."""
    print("🔧 Starting Black Rook Fix Process")
    print("=" * 50)
    
    # First image to fix (from the analysis)
    first_image = "NEW_20250805_135338_005"
    dataset_type = "test"
    
    print(f"🎯 Starting with: {first_image} ({dataset_type.upper()})")
    print(f"   This image generates problematic black rook pieces")
    print(f"   Fixing it will improve all piece types from this image")
    
    # Check if files exist
    image_path = f"grey_background_dataset/images/{dataset_type}/{first_image}.JPG"
    annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{first_image}.json"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    if not os.path.exists(annotation_path):
        print(f"❌ Annotation not found: {annotation_path}")
        return
    
    print(f"✅ Files found:")
    print(f"   Image: {image_path}")
    print(f"   Annotation: {annotation_path}")
    
    # Load current annotation to show current state
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    current_corners = annotation.get('corners', [])
    current_fen = annotation.get('fen', '')
    
    print(f"\n📊 Current state:")
    print(f"   Corners: {current_corners}")
    print(f"   FEN: {current_fen}")
    
    print(f"\n🚀 Ready to start fixing!")
    print(f"   Run: python3 batch_fix_black_rook_issues.py")
    print(f"   This will process all {29} problematic images systematically")
    
    print(f"\n💡 Alternative: Fix just this one image first")
    print(f"   You can also use the existing fixing scripts for individual images")

if __name__ == "__main__":
    main()
