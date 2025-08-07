#!/usr/bin/env python3
"""
Interactive Chess Board Corner Corrector

This tool helps you manually correct corner annotations for chess board images.
It displays the image with current corner points and allows you to click to correct them.
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

class CornerCorrector:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"
        self.images_path = self.dataset_path / "images"
        self.current_image = None
        self.current_corners = []
        self.current_annotation_file = None
        self.corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.selected_corner = 0
        self.window_name = "Chess Board Corner Corrector"
        
    def load_annotation_files(self):
        """Load all annotation files from train, val, and test directories."""
        annotation_files = []
        for split in ["train", "val", "test"]:
            split_path = self.annotations_path / split
            if split_path.exists():
                for json_file in split_path.glob("*.json"):
                    annotation_files.append((split, json_file))
        return annotation_files
    
    def load_annotation(self, split, json_file):
        """Load annotation data from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_file = data.get('image', '')
        corners = data.get('corners', [])
        fen = data.get('fen', '')
        
        # Find the actual image file
        image_path = self.images_path / split / image_file
        if not image_path.exists():
            # Try different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = image_path.with_suffix(ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        return {
            'split': split,
            'json_file': json_file,
            'image_path': image_path,
            'corners': corners,
            'fen': fen,
            'image_file': image_file
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for corner selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Update the selected corner
            self.current_corners[self.selected_corner] = [x, y]
            print(f"Updated {self.corner_names[self.selected_corner]}: ({x}, {y})")
            
            # Move to next corner
            self.selected_corner = (self.selected_corner + 1) % 4
            
            # Redraw the image
            self.draw_image()
    
    def draw_image(self):
        """Draw the image with current corners and instructions."""
        if self.current_image is None:
            return
        
        # Create a copy for drawing
        display_img = self.current_image.copy()
        
        # Draw current corners
        for i, corner in enumerate(self.current_corners):
            if corner:
                color = (0, 255, 0) if i == self.selected_corner else (255, 0, 0)
                cv2.circle(display_img, (corner[0], corner[1]), 10, color, -1)
                cv2.putText(display_img, f"{i+1}", (corner[0]+15, corner[1]+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw corner order guide
        if len(self.current_corners) == 4 and all(self.current_corners):
            # Draw lines connecting corners in order
            for i in range(4):
                pt1 = tuple(self.current_corners[i])
                pt2 = tuple(self.current_corners[(i + 1) % 4])
                cv2.line(display_img, pt1, pt2, (0, 255, 255), 2)
        
        # Add instructions
        instructions = [
            f"Current: {self.corner_names[self.selected_corner]}",
            "Click to set corner position",
            "Press 'n' for next image",
            "Press 'p' for previous image", 
            "Press 's' to save",
            "Press 'r' to reset corners",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(display_img, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        cv2.imshow(self.window_name, display_img)
    
    def save_annotation(self):
        """Save the current annotation back to the JSON file."""
        if not self.current_annotation_file:
            return
        
        # Load current data
        with open(self.current_annotation_file, 'r') as f:
            data = json.load(f)
        
        # Update corners
        data['corners'] = self.current_corners
        
        # Save back to file
        with open(self.current_annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved annotation to {self.current_annotation_file}")
    
    def reset_corners(self):
        """Reset corners to empty state."""
        self.current_corners = [[], [], [], []]
        self.selected_corner = 0
        self.draw_image()
    
    def run(self):
        """Main correction loop."""
        annotation_files = self.load_annotation_files()
        
        if not annotation_files:
            print("No annotation files found!")
            return
        
        print(f"Found {len(annotation_files)} annotation files")
        
        current_index = 0
        
        while current_index < len(annotation_files):
            split, json_file = annotation_files[current_index]
            annotation_data = self.load_annotation(split, json_file)
            
            # Load image
            if not annotation_data['image_path'].exists():
                print(f"Image not found: {annotation_data['image_path']}")
                current_index += 1
                continue
            
            self.current_image = cv2.imread(str(annotation_data['image_path']))
            if self.current_image is None:
                print(f"Failed to load image: {annotation_data['image_path']}")
                current_index += 1
                continue
            
            # Set up current data
            self.current_corners = annotation_data['corners'].copy()
            self.current_annotation_file = json_file
            self.selected_corner = 0
            
            # Create window and set mouse callback
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            print(f"\nProcessing: {annotation_data['image_file']}")
            print(f"FEN: {annotation_data['fen']}")
            print(f"Current corners: {self.current_corners}")
            
            # Draw initial image
            self.draw_image()
            
            # Main interaction loop
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    # Save current annotation
                    self.save_annotation()
                    current_index += 1
                    break
                elif key == ord('p'):
                    # Save current annotation
                    self.save_annotation()
                    current_index = max(0, current_index - 1)
                    break
                elif key == ord('s'):
                    # Save current annotation
                    self.save_annotation()
                elif key == ord('r'):
                    # Reset corners
                    self.reset_corners()
                elif key == ord('1'):
                    self.selected_corner = 0
                    self.draw_image()
                elif key == ord('2'):
                    self.selected_corner = 1
                    self.draw_image()
                elif key == ord('3'):
                    self.selected_corner = 2
                    self.draw_image()
                elif key == ord('4'):
                    self.selected_corner = 3
                    self.draw_image()
            
            cv2.destroyAllWindows()
        
        print("Finished processing all annotations!")

def main():
    parser = argparse.ArgumentParser(description="Interactive Chess Board Corner Corrector")
    parser.add_argument("--dataset", default="grey_background_dataset", 
                       help="Path to dataset directory")
    
    args = parser.parse_args()
    
    corrector = CornerCorrector(args.dataset)
    corrector.run()

if __name__ == "__main__":
    main() 