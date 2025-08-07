#!/usr/bin/env python3
"""
Interactive tool to pick corner coordinates by clicking on chess board images.
This is very useful for updating corner coordinates in annotations.
"""

import cv2
import json
import os
import argparse
from pathlib import Path

class CornerPicker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.corners = []
        self.corner_names = ['Top-Left (a8)', 'Top-Right (h8)', 'Bottom-Right (h1)', 'Bottom-Left (a1)']
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
        
        # Load image
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for display if too large
        height, width = self.original_img.shape[:2]
        max_display_size = 1200
        
        if max(height, width) > max_display_size:
            self.scale = max_display_size / max(height, width)
            self.display_width = int(width * self.scale)
            self.display_height = int(height * self.scale)
            self.display_img = cv2.resize(self.original_img, (self.display_width, self.display_height))
            print(f"📏 Resized for display: {self.display_width}x{self.display_height}")
        else:
            self.scale = 1.0
            self.display_width = width
            self.display_height = height
            self.display_img = self.original_img.copy()
        
        self.current_img = self.display_img.copy()
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                # Convert display coordinates to original image coordinates
                original_x = int(x / self.scale)
                original_y = int(y / self.scale)
                
                self.corners.append([original_x, original_y])
                
                # Draw the corner
                corner_idx = len(self.corners) - 1
                color = self.colors[corner_idx]
                cv2.circle(self.current_img, (x, y), 10, color, -1)
                cv2.circle(self.current_img, (x, y), 10, (255, 255, 255), 2)
                
                # Draw label
                label = f"{corner_idx + 1}: ({original_x},{original_y})"
                cv2.putText(self.current_img, label, (x + 15, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                print(f"🎯 Corner {corner_idx + 1} ({self.corner_names[corner_idx]}): ({original_x}, {original_y})")
                
                # Draw lines connecting corners
                if len(self.corners) > 1:
                    for i in range(len(self.corners) - 1):
                        pt1 = (int(self.corners[i][0] * self.scale), int(self.corners[i][1] * self.scale))
                        pt2 = (int(self.corners[i+1][0] * self.scale), int(self.corners[i+1][1] * self.scale))
                        cv2.line(self.current_img, pt1, pt2, (255, 255, 255), 2)
                
                # Close the board outline
                if len(self.corners) == 4:
                    pt1 = (int(self.corners[3][0] * self.scale), int(self.corners[3][1] * self.scale))
                    pt2 = (int(self.corners[0][0] * self.scale), int(self.corners[0][1] * self.scale))
                    cv2.line(self.current_img, pt1, pt2, (255, 255, 255), 2)
                    print("\n✅ All 4 corners selected!")
                    print("Press 's' to save, 'r' to reset, or 'q' to quit")
    
    def run(self):
        window_name = f"Corner Picker - {Path(self.image_path).name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"📸 Interactive Corner Picker")
        print(f"   Image: {self.image_path}")
        print(f"   Size: {self.original_img.shape[1]}x{self.original_img.shape[0]} pixels")
        print(f"\n🎯 Instructions:")
        print(f"   1. Click on the 4 corners of the chess board in this order:")
        for i, name in enumerate(self.corner_names):
            print(f"      {i+1}. {name}")
        print(f"   2. Press 's' to save coordinates")
        print(f"   3. Press 'r' to reset and start over")
        print(f"   4. Press 'q' to quit without saving")
        print(f"\n   Click on the first corner (Top-Left a8)...")
        
        while True:
            cv2.imshow(window_name, self.current_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("❌ Quitting without saving")
                break
            elif key == ord('r'):
                print("🔄 Resetting corners...")
                self.corners = []
                self.current_img = self.display_img.copy()
            elif key == ord('s') and len(self.corners) == 4:
                print("\n💾 Saving corner coordinates...")
                corner_string = " ".join([f"{x},{y}" for x, y in self.corners])
                print(f"   Corner coordinates: {corner_string}")

                # Save to annotation JSON file
                base_name = Path(self.image_path).stem
                annotation_dir = 'grey_background_dataset/annotations/train'
                os.makedirs(annotation_dir, exist_ok=True)
                annotation_path = os.path.join(annotation_dir, base_name + '.json')

                # Load or create annotation
                if os.path.exists(annotation_path):
                    with open(annotation_path, 'r') as f:
                        annotation = json.load(f)
                else:
                    annotation = {'image': os.path.basename(self.image_path), 'corners': [], 'fen': ''}

                annotation['corners'] = self.corners

                # Write and flush immediately
                with open(annotation_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                print(f"   Updated corners in: {annotation_path} (saved and flushed)")
                print("✅ Corners saved. Closing window and advancing to next image.")
                # Break the loop to close window and exit
                break
        cv2.destroyAllWindows()
        return self.corners if len(self.corners) == 4 else None

def main():
    parser = argparse.ArgumentParser(description="Interactive corner picker for chess board images")
    parser.add_argument("image", help="Path to the chess board image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    try:
        picker = CornerPicker(args.image)
        corners = picker.run()
        
        if corners:
            print(f"\n✅ Successfully picked {len(corners)} corners!")
            print("You can now use these coordinates with the update_corners.py script:")
            corner_string = " ".join([f"{x},{y}" for x, y in corners])
            print(f"python update_corners.py --update {Path(args.image).stem} \"{corner_string}\"")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 