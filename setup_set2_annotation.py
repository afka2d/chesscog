#!/usr/bin/env python3
"""
Setup Script for Set 2 Annotation
=================================

Quick setup script to help you get started with annotating your new chess set images.
"""

import os
import sys
from pathlib import Path
import subprocess
import requests

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if robust API is running
    try:
        response = requests.get("http://localhost:8005/health", timeout=5)
        if response.status_code == 200:
            print("✅ Robust corner detection API is running")
        else:
            print("❌ Robust corner detection API not responding properly")
            return False
    except:
        print("❌ Robust corner detection API not available")
        return False
    
    # Check if we're in the right directory
    if not Path("robust_corner_api.py").exists():
        print("❌ Not in the correct directory. Please run from chesscog root.")
        return False
    
    print("✅ All requirements met")
    return True

def find_images():
    """Find chess images to annotate"""
    print("\n🔍 Looking for chess images...")
    
    # Common locations where images might be
    possible_locations = [
        "~/Desktop/Chess Images",
        "~/Downloads/Chess Images", 
        "~/Pictures/Chess Images",
        "./chess_set2_images",
        "./new_chess_images",
        "./set2_images"
    ]
    
    found_dirs = []
    for location in possible_locations:
        expanded_path = Path(location).expanduser()
        if expanded_path.exists() and expanded_path.is_dir():
            # Check if it contains images
            image_files = list(expanded_path.glob("*.JPG")) + list(expanded_path.glob("*.jpg"))
            if image_files:
                found_dirs.append((expanded_path, len(image_files)))
                print(f"✅ Found {len(image_files)} images in {expanded_path}")
    
    if not found_dirs:
        print("❌ No chess images found in common locations")
        print("Please specify the path to your chess images manually")
        return None
    
    if len(found_dirs) == 1:
        return found_dirs[0][0]
    
    # Multiple directories found, let user choose
    print("\n📁 Multiple image directories found:")
    for i, (path, count) in enumerate(found_dirs):
        print(f"  {i+1}. {path} ({count} images)")
    
    while True:
        try:
            choice = int(input(f"\nChoose directory (1-{len(found_dirs)}): ")) - 1
            if 0 <= choice < len(found_dirs):
                return found_dirs[choice][0]
            else:
                print("Invalid choice")
        except ValueError:
            print("Please enter a number")

def create_output_structure():
    """Create output directory structure"""
    output_dir = Path("./chess_set2_annotations")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "annotations").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    print(f"✅ Created output directory: {output_dir}")
    return output_dir

def show_usage_instructions(images_dir, output_dir):
    """Show usage instructions"""
    print(f"\n🎯 SETUP COMPLETE!")
    print("=" * 60)
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"♟️  Chess set: set2")
    print("=" * 60)
    
    print("\n📋 USAGE OPTIONS:")
    print("\n1️⃣  INTERACTIVE ANNOTATION (Recommended for first time):")
    print(f"   python semi_automated_annotation_tool.py")
    print("   - Process one image at a time")
    print("   - Full control over each step")
    print("   - Good for learning the process")
    
    print("\n2️⃣  BATCH PROCESSING (For many images):")
    print(f"   python batch_annotation_helper.py \"{images_dir}\" --output \"{output_dir}\"")
    print("   - Process multiple images efficiently")
    print("   - Resume capability")
    print("   - Progress tracking")
    
    print("\n3️⃣  BATCH WITH LIMITS (Test with few images first):")
    print(f"   python batch_annotation_helper.py \"{images_dir}\" --output \"{output_dir}\" --max 5")
    print("   - Process only first 5 images")
    print("   - Good for testing the process")
    
    print("\n4️⃣  RESUME BATCH PROCESSING:")
    print(f"   python batch_annotation_helper.py \"{images_dir}\" --output \"{output_dir}\" --start 10")
    print("   - Resume from image 10")
    print("   - Useful if processing was interrupted")
    
    print("\n💡 TIPS:")
    print("   - Start with interactive mode to learn the process")
    print("   - Use batch mode for efficiency once comfortable")
    print("   - The robust corner detection API provides good initial corners")
    print("   - You can always manually adjust corners if needed")
    print("   - FEN format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

def main():
    """Main setup function"""
    print("🚀 CHESS SET 2 ANNOTATION SETUP")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return
    
    # Find images
    images_dir = find_images()
    if not images_dir:
        print("\n❌ No images found. Please place your chess images in a directory and run again.")
        return
    
    # Create output structure
    output_dir = create_output_structure()
    
    # Show usage instructions
    show_usage_instructions(images_dir, output_dir)
    
    print(f"\n🎉 Ready to start annotating! Choose your preferred method above.")

if __name__ == "__main__":
    main()
