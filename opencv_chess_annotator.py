#!/usr/bin/env python3
"""
OpenCV Chess Board Annotation Tool
- Shows chess board image in OpenCV window
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

# Register HEIF opener
register_heif_opener()

class OpenCVChessAnnotator:
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
        self.display_image = None
        self.corner_history = []  # Track corner placement history for undo
        self.excluded_images = set()  # Track excluded images
        self.completed_count = 0  # Count of completed annotations
        
        # Window name
        self.window_name = "Chess Board Annotation Tool"
        
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
                data = json.load(f)
                if isinstance(data, dict) and 'annotations' in data:
                    # New format with metadata
                    self.excluded_images = set(data.get('excluded_images', []))
                    self.completed_count = data.get('completed_count', 0)
                    return data['annotations']
                else:
                    # Old format - just annotations
                    return data
        return {}
    
    def save_annotations(self):
        """Save current annotations with metadata"""
        annotations_file = self.output_dir / "annotations.json"
        data = {
            'annotations': self.annotations,
            'excluded_images': list(self.excluded_images),
            'completed_count': self.completed_count,
            'total_images': len(self.image_files),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
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
            
            # Convert RGB to BGR for API if needed
            if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
                bgr_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = self.current_image
            
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
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for manual corner placement"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.manual_corners is None:
                self.manual_corners = [[0, 0], [0, 0], [0, 0], [0, 0]]
            
            # Save current state for undo
            self.corner_history.append({
                'corners': [corner[:] for corner in self.manual_corners],
                'corner_index': self.corner_index
            })
            
            # Place corner
            self.manual_corners[self.corner_index] = [x, y]
            
            # Move to next corner
            self.corner_index = (self.corner_index + 1) % 4
            
            corner_names = ['TL', 'TR', 'BR', 'BL']
            print(f"📍 Placed {corner_names[self.corner_index]} corner at ({x}, {y})")
            self.update_display()
            
            # If all 4 corners are placed, automatically prompt for FEN and save
            if self.corner_index == 0 and all(corner != [0, 0] for corner in self.manual_corners):
                print("✅ All 4 corners placed! Auto-prompting for FEN...")
                self.auto_save_and_next()
    
    def update_display(self):
        """Update the display with current image and corners"""
        if self.current_image is None:
            return
        
        # Create a copy for display
        self.display_image = self.current_image.copy()
        
        # Draw auto-detected corners in blue
        if self.auto_corners:
            for i, corner in enumerate(self.auto_corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(self.display_image, (x, y), 15, (255, 0, 0), -1)  # Blue circle
                cv2.circle(self.display_image, (x, y), 20, (255, 255, 255), 3)  # White border
                cv2.putText(self.display_image, f'A{i+1}', (x+20, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw manual corners in red
        if self.manual_corners:
            for i, corner in enumerate(self.manual_corners):
                if corner != [0, 0]:  # Only draw placed corners
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(self.display_image, (x, y), 15, (0, 0, 255), -1)  # Red circle
                    cv2.circle(self.display_image, (x, y), 20, (255, 255, 255), 3)  # White border
                    cv2.putText(self.display_image, f'M{i+1}', (x+20, y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw current corner indicator
        corner_names = ['TL', 'TR', 'BR', 'BL']
        cv2.putText(self.display_image, f"Click to place {corner_names[self.corner_index]} corner", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(self.display_image, "Left click: Place corner | 'n': Next | 'p': Previous | 'a': Auto | 'u': Undo | 'x': Exclude | 'q': Quit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show progress information
        progress = self.get_progress_info()
        progress_text = f"Progress: {progress['completed']} completed, {progress['excluded']} excluded, {progress['remaining']} remaining"
        cv2.putText(self.display_image, progress_text, (10, self.display_image.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show image info
        info_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {self.current_image_path.name}"
        cv2.putText(self.display_image, info_text, (10, self.display_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow(self.window_name, self.display_image)
    
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
        self.corner_history = []
        print("🗑️ Cleared manual corners")
        self.update_display()
    
    def undo_last_corner(self):
        """Undo the last corner placement"""
        if self.corner_history:
            # Restore previous state
            prev_state = self.corner_history.pop()
            self.manual_corners = prev_state['corners']
            self.corner_index = prev_state['corner_index']
            
            corner_names = ['TL', 'TR', 'BR', 'BL']
            print(f"↩️ Undid {corner_names[self.corner_index]} corner placement")
            self.update_display()
        else:
            print("❌ Nothing to undo")
    
    def exclude_current_image(self):
        """Exclude current image from training data"""
        if self.current_image_path:
            image_name = self.current_image_path.name
            self.excluded_images.add(image_name)
            print(f"❌ Excluded {image_name} from training data")
            self.save_annotations()
            self.next_image()
        else:
            print("❌ No image to exclude")
    
    def get_progress_info(self):
        """Get progress information"""
        total_images = len(self.image_files)
        excluded_count = len(self.excluded_images)
        completed_count = len(self.annotations)
        remaining = total_images - excluded_count - completed_count
        
        return {
            'total': total_images,
            'completed': completed_count,
            'excluded': excluded_count,
            'remaining': remaining
        }
    
    def find_first_unannotated_image(self):
        """Find the first image that needs annotation"""
        for i, image_path in enumerate(self.image_files):
            image_name = image_path.name
            if (image_name not in self.excluded_images and 
                image_name not in self.annotations):
                self.current_index = i
                return
        
        # If no unannotated images found, set index beyond range
        self.current_index = len(self.image_files)
    
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
    
    def auto_save_and_next(self):
        """Automatically save current annotation and move to next image"""
        # Save current annotation
        self.save_current_annotation()
        
        # Move to next image
        self.next_image()
    
    def next_image(self):
        """Move to next image, skipping excluded and already annotated ones"""
        # Find next non-excluded, non-annotated image
        start_index = self.current_index
        while True:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                image_name = self.image_files[self.current_index].name
                if (image_name not in self.excluded_images and 
                    image_name not in self.annotations):
                    self.load_current_image()
                    return
            else:
                # Reached the end
                print("🎉 Reached the last image!")
                self.save_annotations()
                cv2.destroyAllWindows()
                print("🎉 All images annotated! Annotation complete!")
                return
            
            # Prevent infinite loop if all remaining images are excluded or annotated
            if self.current_index == start_index:
                print("🎉 All remaining images are excluded or already annotated!")
                self.save_annotations()
                cv2.destroyAllWindows()
                print("🎉 All images annotated! Annotation complete!")
                return
    
    def prev_image(self):
        """Move to previous image, skipping excluded and already annotated ones"""
        # Find previous non-excluded, non-annotated image
        start_index = self.current_index
        while True:
            if self.current_index > 0:
                self.current_index -= 1
                image_name = self.image_files[self.current_index].name
                if (image_name not in self.excluded_images and 
                    image_name not in self.annotations):
                    self.load_current_image()
                    return
            else:
                # Reached the beginning
                print("📷 Already at the first image!")
                return
            
            # Prevent infinite loop if all previous images are excluded or annotated
            if self.current_index == start_index:
                print("📷 All previous images are excluded or already annotated!")
                return
    
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
        self.corner_history = []
        
        # Auto-detect corners
        self.auto_corners = self.auto_detect_corners()
        
        # Update display
        self.update_display()
    
    def run(self):
        """Main annotation loop"""
        print("🚀 Starting OpenCV Chess Annotation Tool")
        print("=" * 60)
        print("🎮 CONTROLS:")
        print("   Left click: Place corner (auto-saves when all 4 placed)")
        print("   'n': Next image (auto-saves if corners placed)")
        print("   'p': Previous image")
        print("   'a': Use auto-detected corners")
        print("   'c': Clear manual corners")
        print("   'u': Undo last corner placement")
        print("   'x': Exclude current image (skip horizontal photos)")
        print("   's': Save current annotation")
        print("   'q': Quit and save")
        print("=" * 60)
        
        # Show progress summary
        progress = self.get_progress_info()
        print(f"📊 PROGRESS SUMMARY:")
        print(f"   Total images: {progress['total']}")
        print(f"   Completed: {progress['completed']}")
        print(f"   Excluded: {progress['excluded']}")
        print(f"   Remaining: {progress['remaining']}")
        print("=" * 60)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Load first unannotated image
        self.find_first_unannotated_image()
        if self.current_index < len(self.image_files):
            self.load_current_image()
        else:
            print("🎉 All images are already annotated!")
            return
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # If we have corners placed, save and move to next
                if self.manual_corners and all(corner != [0, 0] for corner in self.manual_corners):
                    print("✅ Corners detected! Auto-prompting for FEN...")
                    self.auto_save_and_next()
                else:
                    self.next_image()
            elif key == ord('p'):
                self.prev_image()
            elif key == ord('a'):
                self.use_auto_corners()
            elif key == ord('c'):
                self.clear_manual_corners()
            elif key == ord('u'):
                self.undo_last_corner()
            elif key == ord('x'):
                self.exclude_current_image()
            elif key == ord('s'):
                self.save_current_annotation()
        
        # Cleanup
        cv2.destroyAllWindows()
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
    annotator = OpenCVChessAnnotator(image_dir, output_dir, chess_set)
    
    # Run annotation
    annotator.run()

if __name__ == "__main__":
    main()
