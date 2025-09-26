#!/usr/bin/env python3
"""
Improved Chess Board Annotation Tool
- Navigation between images
- Auto-detection of corners using API
- FEN input with default starting position
- Automatic saving
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
import requests
import time

# Register HEIF opener
register_heif_opener()

class ImprovedChessAnnotator:
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
        
        # Load existing annotations
        self.annotations = self.load_annotations()
        
        # API endpoint for corner detection
        self.api_url = "http://localhost:8005/detect_corners"
        
        print(f"📁 Image directory: {self.image_dir}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🏷️  Chess set: {self.chess_set}")
        print(f"🔗 API endpoint: {self.api_url}")
        print("=" * 60)
    
    def find_images(self):
        """Find all supported image files"""
        extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif']
        images = []
        
        for ext in extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def load_annotations(self):
        """Load existing annotations"""
        annotations_file = self.output_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_annotations(self):
        """Save current annotations"""
        annotations_file = self.output_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"💾 Annotations saved to {annotations_file}")
    
    def load_image(self, image_path):
        """Load and display an image"""
        try:
            self.current_image_path = image_path
            
            # Handle HEIC files
            if image_path.suffix.lower() == '.heic':
                pil_img = Image.open(image_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                self.current_image = np.array(pil_img)
                print(f"✅ Loaded HEIC image: {image_path.name} ({self.current_image.shape})")
            else:
                self.current_image = cv2.imread(str(image_path))
                if self.current_image is None:
                    print(f"❌ Could not load image: {image_path}")
                    return False
                # Convert BGR to RGB for display
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                print(f"✅ Loaded image: {image_path.name} ({self.current_image.shape})")
            
            return True
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
    
    def auto_detect_corners(self):
        """Use API to auto-detect corners"""
        try:
            print("🔍 Auto-detecting corners using API...")
            
            # Convert image to bytes for API
            if self.current_image is None:
                print("❌ No image loaded")
                return None
            
            # Convert RGB back to BGR for API
            bgr_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', bgr_image)
            image_bytes = buffer.tobytes()
            
            # Make API request
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(self.api_url, files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                corners = result.get('corners', [])
                confidence = result.get('confidence', 0.0)
                
                if corners and len(corners) == 4:
                    print(f"✅ Auto-detected corners (confidence: {confidence:.3f})")
                    return corners
                else:
                    print("❌ API returned invalid corners")
                    return None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Auto-detection failed: {e}")
            return None
    
    def get_default_fen(self):
        """Get default FEN for starting position from white side"""
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def annotate_image(self):
        """Annotate current image"""
        image_name = self.current_image_path.name
        
        print(f"\n📸 Annotating: {image_name}")
        print(f"📊 Progress: {self.current_index + 1}/{len(self.image_files)}")
        
        # Check if already annotated
        if image_name in self.annotations:
            print("ℹ️  Image already annotated, showing existing data...")
            existing = self.annotations[image_name]
            print(f"   Corners: {existing.get('corners', 'Not set')}")
            print(f"   FEN: {existing.get('fen', 'Not set')}")
            print(f"   Chess set: {existing.get('chess_set', 'Not set')}")
        
        # Auto-detect corners
        auto_corners = self.auto_detect_corners()
        
        if auto_corners:
            print(f"🎯 Auto-detected corners: {auto_corners}")
            
            # Ask if user wants to use auto-detected corners
            use_auto = input("Use auto-detected corners? (y/n): ").lower().strip()
            if use_auto == 'y':
                corners = auto_corners
                print("✅ Using auto-detected corners")
            else:
                corners = None
                print("❌ Skipping auto-detected corners")
        else:
            corners = None
            print("❌ No auto-detected corners available")
        
        # Get FEN input
        default_fen = self.get_default_fen()
        print(f"\n📝 Enter FEN (default: {default_fen})")
        fen_input = input("FEN: ").strip()
        
        if not fen_input:
            fen_input = default_fen
            print(f"✅ Using default FEN: {fen_input}")
        
        # Save annotation
        self.annotations[image_name] = {
            'corners': corners,
            'fen': fen_input,
            'chess_set': self.chess_set,
            'auto_detected_corners': auto_corners,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Annotation saved for {image_name}")
        return True
    
    def show_menu(self):
        """Show navigation menu"""
        print("\n" + "=" * 60)
        print("🎮 NAVIGATION MENU")
        print("=" * 60)
        print("n/next     - Next image")
        print("p/prev     - Previous image")
        print("j/jump     - Jump to specific image number")
        print("s/save     - Save annotations")
        print("q/quit     - Quit and save")
        print("h/help     - Show this menu")
        print("=" * 60)
    
    def run(self):
        """Main annotation loop"""
        print("🚀 Starting Improved Chess Annotation Tool")
        print("=" * 60)
        
        while self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            
            # Load image
            if not self.load_image(current_image):
                print(f"❌ Failed to load {current_image.name}, skipping...")
                self.current_index += 1
                continue
            
            # Annotate image
            self.annotate_image()
            
            # Show menu and get user input
            self.show_menu()
            choice = input(f"\nImage {self.current_index + 1}/{len(self.image_files)} - Choose action: ").lower().strip()
            
            if choice in ['n', 'next']:
                self.current_index += 1
            elif choice in ['p', 'prev']:
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("❌ Already at first image")
            elif choice in ['j', 'jump']:
                try:
                    new_index = int(input(f"Enter image number (1-{len(self.image_files)}): ")) - 1
                    if 0 <= new_index < len(self.image_files):
                        self.current_index = new_index
                    else:
                        print("❌ Invalid image number")
                except ValueError:
                    print("❌ Please enter a valid number")
            elif choice in ['s', 'save']:
                self.save_annotations()
            elif choice in ['q', 'quit']:
                break
            elif choice in ['h', 'help']:
                continue
            else:
                print("❌ Invalid choice, try again")
        
        # Save final annotations
        self.save_annotations()
        print("\n🎉 Annotation complete!")
        print(f"📊 Total images processed: {len(self.annotations)}")
        print(f"💾 Annotations saved to: {self.output_dir}")

def main():
    # Configuration
    image_dir = "/Users/tonyblum/Desktop/marshall photos"
    output_dir = "./marshall_chess_annotations"
    chess_set = "marshall"
    
    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"❌ Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Create annotator
    annotator = ImprovedChessAnnotator(image_dir, output_dir, chess_set)
    
    # Run annotation
    annotator.run()

if __name__ == "__main__":
    main()
