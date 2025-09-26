#!/usr/bin/env python3
"""
Simple Chess Board Annotation Tool
A simplified version that should work reliably on macOS
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener
register_heif_opener()

class SimpleChessAnnotator:
    def __init__(self, image_dir, output_dir, chess_set="marshall"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.chess_set = chess_set
        self.current_index = 0
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Find all images
        self.image_files = self.find_images()
        print(f"🔍 Found {len(self.image_files)} images")
        
        if not self.image_files:
            print("❌ No images found!")
            sys.exit(1)
    
    def find_images(self):
        """Find all supported image files"""
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.HEIC'}
        images = []
        
        for ext in extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def load_image(self, image_path):
        """Load image using OpenCV"""
        try:
            if image_path.suffix.lower() == '.heic':
                # Use PIL for HEIC, then convert to OpenCV format
                pil_img = Image.open(image_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                # Convert PIL to OpenCV format (RGB -> BGR)
                img_array = np.array(pil_img)
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(str(image_path))
            
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
                
            print(f"✅ Loaded image: {image_path.name} ({img.shape})")
            return img
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def annotate_image(self, image_path):
        """Annotate a single image"""
        print(f"\n📸 Annotating: {image_path.name}")
        
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return None
        
        # Resize for display if too large
        display_img = img.copy()
        height, width = img.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_img = cv2.resize(img, (new_width, new_height))
            scale_factor = scale
        else:
            scale_factor = 1.0
        
        print(f"🖼️  Image size: {width}x{height} (display: {display_img.shape[1]}x{display_img.shape[0]})")
        
        # Get corners from user
        corners = self.get_corners_interactive(display_img, image_path.name)
        if corners is None:
            return None
        
        # Scale corners back to original image size
        if scale_factor != 1.0:
            corners = [(int(x/scale_factor), int(y/scale_factor)) for x, y in corners]
        
        # Get FEN from user
        fen = self.get_fen_from_user()
        if fen is None:
            return None
        
        # Create annotation
        annotation = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "chess_set": self.chess_set,
            "corners": corners,
            "fen": fen,
            "timestamp": str(Path(image_path).stat().st_mtime)
        }
        
        return annotation
    
    def get_corners_interactive(self, img, image_name):
        """Get corners using OpenCV interactive window"""
        print(f"\n🎯 Click on the 4 corners of the chess board in order:")
        print("   1. Top-Left (TL)")
        print("   2. Top-Right (TR)")  
        print("   3. Bottom-Right (BR)")
        print("   4. Bottom-Left (BL)")
        print("   Press 'r' to reset, 'q' to quit")
        
        corners = []
        img_copy = img.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal corners, img_copy
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(corners) < 4:
                    corners.append((x, y))
                    print(f"   Corner {len(corners)}: ({x}, {y})")
                    
                    # Draw the corner
                    cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(img_copy, (x, y), 12, (255, 255, 255), 2)
                    
                    # Draw corner number
                    cv2.putText(img_copy, str(len(corners)), (x+15, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Draw lines between corners
                    if len(corners) > 1:
                        cv2.line(img_copy, corners[-2], corners[-1], (0, 255, 0), 2)
                    if len(corners) == 4:
                        # Close the quadrilateral
                        cv2.line(img_copy, corners[3], corners[0], (0, 255, 0), 2)
                        print("   ✅ All 4 corners selected!")
                
                cv2.imshow(f"Chess Annotation - {image_name}", img_copy)
        
        cv2.namedWindow(f"Chess Annotation - {image_name}", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(f"Chess Annotation - {image_name}", mouse_callback)
        cv2.imshow(f"Chess Annotation - {image_name}", img_copy)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):
                corners = []
                img_copy = img.copy()
                cv2.imshow(f"Chess Annotation - {image_name}", img_copy)
                print("   🔄 Reset - click corners again")
            elif len(corners) == 4:
                # All corners selected, wait for confirmation
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return None
                elif key == ord('r'):
                    corners = []
                    img_copy = img.copy()
                    cv2.imshow(f"Chess Annotation - {image_name}", img_copy)
                    print("   🔄 Reset - click corners again")
                elif key == 13 or key == 32:  # Enter or Space
                    break
        
        cv2.destroyAllWindows()
        return corners
    
    def get_fen_from_user(self):
        """Get FEN string from user"""
        print(f"\n♟️  Enter the FEN for this position:")
        print("   (or press Enter to skip this image)")
        
        fen = input("FEN: ").strip()
        
        if not fen:
            print("   ⏭️  Skipped FEN entry")
            return None
        
        # Basic FEN validation
        if len(fen.split()) < 6:
            print("   ⚠️  Invalid FEN format, but saving anyway")
        
        return fen
    
    def save_annotation(self, annotation):
        """Save annotation to JSON file"""
        if annotation is None:
            return
        
        output_file = self.output_dir / f"{annotation['image_name']}.json"
        
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"💾 Saved annotation: {output_file}")
    
    def run(self):
        """Run the annotation process"""
        print(f"\n🎮 CHESS ANNOTATION TOOL")
        print(f"📁 Images: {len(self.image_files)}")
        print(f"♟️  Chess set: {self.chess_set}")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🎯 Controls:")
        print(f"   • Click to place corners (TL, TR, BR, BL)")
        print(f"   • 'r' to reset corners")
        print(f"   • 'q' to quit")
        print(f"   • Enter FEN when prompted")
        print(f"   • Press Enter to skip an image")
        
        for i, image_path in enumerate(self.image_files):
            print(f"\n{'='*60}")
            print(f"📸 Image {i+1}/{len(self.image_files)}: {image_path.name}")
            
            annotation = self.annotate_image(image_path)
            self.save_annotation(annotation)
            
            if annotation is None:
                print("   ⏭️  Skipped this image")
                continue
            
            # Ask if user wants to continue
            if i < len(self.image_files) - 1:
                response = input(f"\nContinue to next image? (y/n/q): ").strip().lower()
                if response == 'q':
                    print("   👋 Quitting annotation process")
                    break
                elif response == 'n':
                    print("   ⏸️  Stopping annotation process")
                    break
        
        print(f"\n✅ Annotation complete!")
        print(f"📁 Saved annotations in: {self.output_dir}")

def main():
    # Configuration
    image_dir = "/Users/tonyblum/Desktop/marshall photos"
    output_dir = "./marshall_chess_annotations"
    chess_set = "marshall"
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Create annotator and run
    annotator = SimpleChessAnnotator(image_dir, output_dir, chess_set)
    annotator.run()

if __name__ == "__main__":
    main()
