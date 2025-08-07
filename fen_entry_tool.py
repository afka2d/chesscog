#!/usr/bin/env python3
"""
Interactive FEN Entry Tool

This tool helps you systematically add FEN strings to chess board images.
It displays each image that needs a FEN string and allows you to enter it.
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import glob

class FENEntryTool:
    def __init__(self, dataset_path="grey_background_dataset"):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"
        self.images_path = self.dataset_path / "images"
        self.current_image = None
        self.current_annotation = None
        self.current_file = None
        
        # Find all files that need FEN strings
        self.files_needing_fen = self.find_files_needing_fen()
        self.current_index = 0
        
        print(f"Found {len(self.files_needing_fen)} images that need FEN strings")
        
    def find_files_needing_fen(self):
        """Find all annotation files that still have the default empty FEN."""
        files = []
        pattern = str(self.annotations_path / "**" / "*.json")
        for json_file in glob.glob(pattern, recursive=True):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('fen') == "8/8/8/8/8/8/8/8 w - - 0 1":
                        files.append(json_file)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        return sorted(files)
    
    def load_annotation(self, json_file):
        """Load annotation data from JSON file."""
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None
    
    def save_annotation(self, json_file, data):
        """Save annotation data to JSON file."""
        try:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {json_file}: {e}")
            return False
    
    def find_image_file(self, image_name):
        """Find the image file in train/val/test directories."""
        for subdir in ['train', 'val', 'test']:
            image_path = self.images_path / subdir / image_name
            if image_path.exists():
                return str(image_path)
        return None
    
    def display_image_with_info(self, json_file):
        """Display the image with current annotation info."""
        annotation = self.load_annotation(json_file)
        if not annotation:
            return False
            
        image_name = annotation.get('image', '')
        image_path = self.find_image_file(image_name)
        
        if not image_path:
            print(f"Image not found: {image_name}")
            return False
            
        # Load and display image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
            
        # Resize image to fit screen better first
        corners = annotation.get('corners', [])
        height, width = image.shape[:2]
        max_height = 800
        scale = 1.0
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, max_height))
        
        # Draw corner points if they exist (using scaled coordinates)
        if len(corners) == 4 and all(len(c) == 2 for c in corners):
            # Draw corners with scaled coordinates
            for i, (x, y) in enumerate(corners):
                scaled_x = int(x * scale)
                scaled_y = int(y * scale)
                cv2.circle(image, (scaled_x, scaled_y), 15, (0, 255, 0), -1)
                cv2.putText(image, str(i+1), (scaled_x-10, scaled_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw lines connecting scaled corners
            scaled_corners = [[int(x * scale), int(y * scale)] for x, y in corners]
            pts = np.array(scaled_corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (255, 255, 0), 3)
        
        # Display image
        window_name = f"FEN Entry Tool - {image_name} ({self.current_index + 1}/{len(self.files_needing_fen)})"
        cv2.imshow(window_name, image)
        cv2.moveWindow(window_name, 100, 100)
        
        # Print current info
        print(f"\n{'='*60}")
        print(f"Image: {image_name}")
        print(f"File: {json_file}")
        print(f"Progress: {self.current_index + 1}/{len(self.files_needing_fen)}")
        print(f"Current FEN: {annotation.get('fen', 'None')}")
        print(f"{'='*60}")
        
        self.current_annotation = annotation
        self.current_file = json_file
        return True
    
    def get_fen_input(self):
        """Get FEN string input from user."""
        print("\nEnter the FEN string for this chess position:")
        print("(Or type 'skip' to skip this image, 'quit' to exit)")
        print("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        while True:
            fen_input = input("\nFEN: ").strip()
            
            if fen_input.lower() == 'quit':
                return 'quit'
            elif fen_input.lower() == 'skip':
                return 'skip'
            elif len(fen_input) == 0:
                print("Please enter a FEN string, 'skip', or 'quit'")
                continue
            else:
                # Basic FEN validation (should have 6 parts)
                parts = fen_input.split()
                if len(parts) != 6:
                    print("Invalid FEN format. Should have 6 parts separated by spaces.")
                    print("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                    continue
                return fen_input
    
    def process_images(self):
        """Main processing loop."""
        if not self.files_needing_fen:
            print("No images need FEN strings!")
            return
            
        print(f"\nStarting FEN entry for {len(self.files_needing_fen)} images")
        print("\nControls:")
        print("- Enter FEN string when prompted")
        print("- Type 'skip' to skip current image")
        print("- Type 'quit' to exit")
        print("- Press any key in image window to continue to input")
        
        while self.current_index < len(self.files_needing_fen):
            json_file = self.files_needing_fen[self.current_index]
            
            # Display the image
            if not self.display_image_with_info(json_file):
                print(f"Skipping {json_file} due to error")
                self.current_index += 1
                continue
            
            # Wait for key press to continue
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Get FEN input
            fen_result = self.get_fen_input()
            
            if fen_result == 'quit':
                print(f"\nExiting. Progress saved up to image {self.current_index}")
                break
            elif fen_result == 'skip':
                print("Skipping this image")
            else:
                # Save the FEN string
                self.current_annotation['fen'] = fen_result
                if self.save_annotation(self.current_file, self.current_annotation):
                    print(f"✅ Saved FEN for {self.current_annotation['image']}")
                else:
                    print(f"❌ Failed to save FEN for {self.current_annotation['image']}")
            
            self.current_index += 1
        
        cv2.destroyAllWindows()
        print(f"\nCompleted! Processed {self.current_index}/{len(self.files_needing_fen)} images")

def main():
    parser = argparse.ArgumentParser(description='Interactive FEN Entry Tool for Chess Images')
    parser.add_argument('--dataset', default='grey_background_dataset',
                       help='Path to dataset directory (default: grey_background_dataset)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory '{args.dataset}' not found")
        sys.exit(1)
    
    # Create and run the tool
    tool = FENEntryTool(args.dataset)
    tool.process_images()

if __name__ == "__main__":
    main()