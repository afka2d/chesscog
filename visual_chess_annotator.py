#!/usr/bin/env python3
"""
Visual Chess Board Annotation Tool
- Shows chess board image in a popup window
- Auto-detects corners using API for comparison
- Allows manual corner adjustment by clicking
- FEN input with default starting position
- Navigation between images
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Register HEIF opener
register_heif_opener()

class VisualChessAnnotator:
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
        
        # Current state
        self.current_image = None
        self.current_image_path = None
        self.auto_corners = None
        self.manual_corners = None
        self.corner_index = 0  # 0=TL, 1=TR, 2=BR, 3=BL
        
        # Matplotlib setup
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle("Chess Board Annotation Tool", fontsize=16)
        
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
    
    def on_click(self, event):
        """Handle mouse clicks for manual corner placement"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if self.manual_corners is None:
                self.manual_corners = [[0, 0], [0, 0], [0, 0], [0, 0]]
            
            # Place corner
            self.manual_corners[self.corner_index] = [x, y]
            
            # Move to next corner
            self.corner_index = (self.corner_index + 1) % 4
            
            print(f"📍 Placed corner {self.corner_index} at ({x}, {y})")
            self.update_display()
    
    def update_display(self):
        """Update the display with current image and corners"""
        self.ax.clear()
        
        if self.current_image is not None:
            self.ax.imshow(self.current_image)
        
        # Draw auto-detected corners in blue
        if self.auto_corners:
            for i, corner in enumerate(self.auto_corners):
                x, y = corner
                self.ax.plot(x, y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
                self.ax.text(x+10, y+10, f'A{i+1}', color='blue', fontsize=12, fontweight='bold')
        
        # Draw manual corners in red
        if self.manual_corners:
            for i, corner in enumerate(self.manual_corners):
                if corner != [0, 0]:  # Only draw placed corners
                    x, y = corner
                    self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                    self.ax.text(x+10, y+10, f'M{i+1}', color='red', fontsize=12, fontweight='bold')
        
        # Draw current corner indicator
        corner_names = ['TL', 'TR', 'BR', 'BL']
        self.ax.set_title(f"Click to place {corner_names[self.corner_index]} corner (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
        
        # Add instructions
        self.ax.text(10, 30, "Left click: Place corner | Right click: Next image | Middle click: Previous image", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, color='black')
        
        self.ax.text(10, self.current_image.shape[0] - 30, 
                    f"Image {self.current_index + 1}/{len(self.image_files)}: {self.current_image_path.name}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    fontsize=10, color='black')
        
        self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'n' or event.key == 'next':
            self.next_image()
        elif event.key == 'p' or event.key == 'prev':
            self.prev_image()
        elif event.key == 'a' or event.key == 'auto':
            self.use_auto_corners()
        elif event.key == 'c' or event.key == 'clear':
            self.clear_manual_corners()
        elif event.key == 's' or event.key == 'save':
            self.save_current_annotation()
        elif event.key == 'q' or event.key == 'quit':
            self.quit()
    
    def next_image(self):
        """Move to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def prev_image(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def use_auto_corners(self):
        """Use auto-detected corners"""
        if self.auto_corners:
            self.manual_corners = self.auto_corners.copy()
            self.corner_index = 0
            print("✅ Using auto-detected corners")
            self.update_display()
        else:
            print("❌ No auto-detected corners available")
    
    def clear_manual_corners(self):
        """Clear manual corners"""
        self.manual_corners = None
        self.corner_index = 0
        print("🗑️ Cleared manual corners")
        self.update_display()
    
    def save_current_annotation(self):
        """Save current annotation"""
        if not self.current_image_path:
            print("❌ No image loaded")
            return
        
        # Get FEN input
        default_fen = self.get_default_fen()
        print(f"\n📝 Enter FEN (default: {default_fen})")
        fen_input = input("FEN: ").strip()
        
        if not fen_input:
            fen_input = default_fen
            print(f"✅ Using default FEN: {fen_input}")
        
        # Use manual corners if available, otherwise auto corners
        corners = self.manual_corners if self.manual_corners else self.auto_corners
        
        # Save annotation
        image_name = self.current_image_path.name
        self.annotations[image_name] = {
            'corners': corners,
            'fen': fen_input,
            'chess_set': self.chess_set,
            'auto_detected_corners': self.auto_corners,
            'manual_corners': self.manual_corners,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Annotation saved for {image_name}")
        self.save_annotations()
    
    def quit(self):
        """Quit the application"""
        self.save_annotations()
        print("\n🎉 Annotation complete!")
        print(f"📊 Total images processed: {len(self.annotations)}")
        print(f"💾 Annotations saved to: {self.output_dir}")
        plt.close('all')
        sys.exit(0)
    
    def load_current_image(self):
        """Load and display current image"""
        current_image = self.image_files[self.current_index]
        
        if not self.load_image(current_image):
            print(f"❌ Failed to load {current_image.name}, skipping...")
            return
        
        # Reset state
        self.auto_corners = None
        self.manual_corners = None
        self.corner_index = 0
        
        # Auto-detect corners
        self.auto_corners = self.auto_detect_corners()
        
        # Update display
        self.update_display()
    
    def run(self):
        """Main annotation loop"""
        print("🚀 Starting Visual Chess Annotation Tool")
        print("=" * 60)
        print("🎮 CONTROLS:")
        print("   Left click: Place corner")
        print("   Right click: Next image")
        print("   Middle click: Previous image")
        print("   'a' or 'auto': Use auto-detected corners")
        print("   'c' or 'clear': Clear manual corners")
        print("   's' or 'save': Save current annotation")
        print("   'q' or 'quit': Quit and save")
        print("=" * 60)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Load first image
        self.load_current_image()
        
        # Show the window
        plt.show(block=True)

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
    annotator = VisualChessAnnotator(image_dir, output_dir, chess_set)
    
    # Run annotation
    annotator.run()

if __name__ == "__main__":
    main()
