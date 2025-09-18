#!/usr/bin/env python3
"""
Setup script for comprehensive accuracy evaluation.
This will guide you through the process step by step.
"""

import os
import sys
import requests
from pathlib import Path

def check_api():
    """Check if local API is running"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Local API is running")
            return True
        else:
            print("❌ Local API not responding correctly")
            return False
    except:
        print("❌ Cannot connect to local API")
        return False

def check_annotations():
    """Check if ground truth annotations exist"""
    dataset_path = "my_chess_images/train/images"
    annotation_files = list(Path(dataset_path).glob("**/*.json"))
    
    if annotation_files:
        print(f"✅ Found {len(annotation_files)} annotation files")
        return True
    else:
        print("❌ No ground truth annotations found")
        return False

def main():
    """Main setup function"""
    print("Chess Model Accuracy Evaluation Setup")
    print("=" * 50)
    
    # Step 1: Check API
    print("\n1. Checking local API...")
    if not check_api():
        print("\nPlease start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Step 2: Check annotations
    print("\n2. Checking ground truth annotations...")
    if not check_annotations():
        print("\nYou need to create ground truth annotations first.")
        print("Choose an option:")
        print("1. Create annotations interactively (recommended)")
        print("2. Create annotations manually")
        print("3. Skip and run evaluation anyway")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\nStarting interactive annotation tool...")
            print("This will open a visual interface where you can click on squares to annotate pieces.")
            print("Follow the instructions in the popup window.")
            
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                os.system("python create_ground_truth_interactive.py")
            else:
                print("Annotation cancelled.")
                return
        elif choice == "2":
            print("\nManual annotation instructions:")
            print("1. Create JSON files for each image")
            print("2. Format: {'a8': {'occupied': True, 'color': 'black', 'piece': 'rook'}, ...}")
            print("3. Save as IMAGE_NAME.json in the same directory as the image")
            return
        elif choice == "3":
            print("Skipping annotation check...")
        else:
            print("Invalid choice. Exiting.")
            return
    
    # Step 3: Run evaluation
    print("\n3. Running comprehensive accuracy evaluation...")
    print("This will measure:")
    print("  - % of squares where occupancy is correct")
    print("  - % of occupied squares where color is correct")
    print("  - % of occupied squares where piece is correct")
    print("  - % of images where entire FEN is 100% correct")
    
    confirm = input("\nContinue with evaluation? (y/n): ").strip().lower()
    if confirm == 'y':
        os.system("python comprehensive_accuracy_evaluation.py")
    else:
        print("Evaluation cancelled.")

if __name__ == "__main__":
    main()
