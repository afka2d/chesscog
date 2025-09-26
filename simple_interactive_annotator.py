#!/usr/bin/env python3
"""
Simple Interactive Chess Annotation Tool
Shows images on screen and allows manual corner adjustment
"""

import cv2
import json
import numpy as np
from pathlib import Path
import requests
import chess
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import os

class SimpleChessAnnotator:
    def __init__(self, image_dir, output_dir, chess_set="set2"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        self.chess_set = chess_set
        
        # Current state
        self.current_image_path = None
        self.current_image = None
        self.current_corners = None
        self.current_fen = None
        self.image_files = []
        self.current_index = 0
        
        # Matplotlib setup
        self.fig = None
        self.ax = None
        
        # Load all chess images
        self.load_chess_images()
        
    def load_chess_images(self):
        """Load all chess images from Downloads"""
        print("🔍 Finding chess images...")
        
        # Find all image files
        image_extensions = ['*.JPG', '*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        for ext in image_extensions:
            self.image_files.extend(self.image_dir.glob(ext))
        
        # Filter out non-chess images (like ChatGPT screenshots)
        chess_images = []
        for img_path in self.image_files:
            if not any(skip in img_path.name.lower() for skip in ['chatgpt', 'screenshot', 'screen']):
                chess_images.append(img_path)
        
        self.image_files = sorted(chess_images)
        print(f"📁 Found {len(self.image_files)} chess images")
        
        if len(self.image_files) == 0:
            print("❌ No chess images found. Please check your Downloads folder.")
            return False
        return True
    
    def load_image(self, image_path):
        """Load and display an image"""
        try:
            self.current_image_path = image_path
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                print(f"❌ Could not load image: {image_path}")
                return False
            
            # Convert BGR to RGB for matplotlib
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded image: {image_path.name} ({self.current_image.shape})")
            return True
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
    
    def detect_corners_automatically(self):
        """Use the robust corner detection API to get initial corners"""
        try:
            print("🔍 Detecting corners automatically...")
            with open(self.current_image_path, 'rb') as f:
                response = requests.post(
                    "http://localhost:8005/detect_corners",
                    files={'file': f},
                    params={'time_budget': 2.0},
                    timeout=10
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    corners = data['corners']
                    print(f"✅ Auto-detected corners: {corners}")
                    return corners
                else:
                    print(f"❌ Auto-detection failed: {data}")
                    return None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error calling API: {e}")
            return None
    
    def setup_matplotlib(self):
        """Setup matplotlib figure and axes"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.fig.suptitle(f"Chess Annotation Tool - Image {self.current_index + 1}/{len(self.image_files)}", fontsize=16)
        
        # Connect mouse click events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add buttons
        ax_accept = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_reject = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_auto = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_clear = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_prev = plt.axes([0.32, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.43, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.54, 0.02, 0.1, 0.04])
        
        self.btn_accept = Button(ax_accept, 'Accept')
        self.btn_reject = Button(ax_reject, 'Skip')
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_quit = Button(ax_quit, 'Quit')
        
        self.btn_accept.on_clicked(self.accept_corners)
        self.btn_reject.on_clicked(self.reject_image)
        self.btn_auto.on_clicked(self.auto_detect)
        self.btn_clear.on_clicked(self.clear_corners)
        self.btn_prev.on_clicked(self.previous_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_quit.on_clicked(self.quit_annotation)
    
    def display_image(self):
        """Display the current image with corners"""
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"{self.current_image_path.name}")
        
        # Draw corners if they exist
        if self.current_corners is not None:
            self.draw_corners()
        
        # Instructions
        self.ax.text(10, 30, "Click to place corners (TL, TR, BR, BL)", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=12, color='black')
        
        self.ax.text(10, self.current_image.shape[0] - 30, 
                    f"Image {self.current_index + 1}/{len(self.image_files)}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    fontsize=12, color='black')
        
        # Show corner status
        if self.current_corners is not None:
            corner_text = f"Corners: {len(self.current_corners)}/4"
            self.ax.text(self.current_image.shape[1] - 150, 30, corner_text,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                        fontsize=12, color='black')
        
        self.fig.canvas.draw()
    
    def draw_corners(self):
        """Draw the current corners on the image"""
        if self.current_corners is None:
            return
        
        corners = np.array(self.current_corners)
        colors = ['red', 'green', 'blue', 'yellow']
        labels = ['TL', 'TR', 'BR', 'BL']
        
        # Draw corner points
        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = corner
            self.ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=3)
            self.ax.text(x + 15, y + 15, label, color=color, fontsize=14, fontweight='bold')
        
        # Draw quadrilateral
        if len(corners) == 4:
            corners_closed = np.vstack([corners, corners[0]])  # Close the polygon
            self.ax.plot(corners_closed[:, 0], corners_closed[:, 1], 'cyan', linewidth=4, alpha=0.8)
    
    def on_click(self, event):
        """Handle mouse clicks to place corners"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if self.current_corners is None:
                self.current_corners = []
            
            if len(self.current_corners) < 4:
                self.current_corners.append([x, y])
                print(f"📍 Placed corner {len(self.current_corners)}: ({x}, {y})")
                self.display_image()
            else:
                print("⚠️  All 4 corners already placed. Click 'Clear' to start over.")
    
    def auto_detect(self, event):
        """Auto-detect corners using the API"""
        corners = self.detect_corners_automatically()
        if corners is not None:
            self.current_corners = corners
            self.display_image()
    
    def clear_corners(self, event):
        """Clear all corners"""
        self.current_corners = None
        self.display_image()
        print("🗑️  Cleared all corners")
    
    def accept_corners(self, event):
        """Accept current corners and proceed to FEN input"""
        if self.current_corners is None or len(self.current_corners) != 4:
            print("⚠️  Please place all 4 corners first")
            return
        
        print(f"✅ Corners accepted: {self.current_corners}")
        self.get_fen_input()
    
    def reject_image(self, event):
        """Skip this image"""
        print("⏭️  Skipping this image")
        self.next_image()
    
    def quit_annotation(self, event):
        """Quit the annotation tool"""
        print("👋 Quitting annotation tool")
        plt.close('all')
    
    def get_fen_input(self):
        """Get FEN input from user"""
        plt.close(self.fig)
        
        print(f"\n♟️  Enter FEN position for {self.current_image_path.name}:")
        print("Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print("Or press Enter to skip this image")
        
        fen_input = input("FEN: ").strip()
        
        if not fen_input:
            print("⏭️  Skipping this image")
            self.next_image()
            return
        
        # Add default ending if not provided
        if ' ' not in fen_input:
            fen_input += " w KQkq - 0 1"
        
        # Validate FEN
        try:
            chess.Board(fen_input)
            print("✅ FEN is valid")
            self.current_fen = fen_input
            self.save_annotation()
        except Exception as e:
            print(f"❌ Invalid FEN: {e}")
            print("Please try again or press Enter to skip")
            self.get_fen_input()
    
    def save_annotation(self):
        """Save the current annotation"""
        annotation = {
            "image_path": str(self.current_image_path),
            "image_name": self.current_image_path.name,
            "chess_set": self.chess_set,
            "corners": self.current_corners,
            "fen": self.current_fen,
            "annotation_method": "interactive_manual",
            "corner_detection_api": "robust_port_8005",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save annotation
        annotation_file = self.output_dir / "annotations" / f"{self.current_image_path.stem}.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save visualization
        vis_img = self.current_image.copy()
        corners_np = np.array(self.current_corners, dtype=np.int32)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
            x, y = corner
            cv2.circle(vis_img, (x, y), 15, color, -1)
            cv2.circle(vis_img, (x, y), 20, (255, 255, 255), 3)
            cv2.putText(vis_img, f'{label}', (x-20, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        
        vis_file = self.output_dir / "visualizations" / f"{self.current_image_path.stem}_corners.jpg"
        cv2.imwrite(str(vis_file), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Annotation saved: {annotation_file}")
        print(f"📊 Visualization saved: {vis_file}")
        
        self.next_image()
    
    def previous_image(self, event=None):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_and_display_current()
    
    def next_image(self, event=None):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_and_display_current()
        else:
            print("🎉 All images processed!")
            plt.close('all')
    
    def load_and_display_current(self):
        """Load and display the current image"""
        if self.current_index >= len(self.image_files):
            print("🎉 All images processed!")
            return
        
        image_path = self.image_files[self.current_index]
        
        # Close previous figure
        if self.fig is not None:
            plt.close(self.fig)
        
        # Load image
        if not self.load_image(image_path):
            self.next_image()
            return
        
        # Reset state
        self.current_corners = None
        self.current_fen = None
        
        # Setup and display
        self.setup_matplotlib()
        self.display_image()
        
        # Try auto-detection first
        self.auto_detect(None)
    
    def run(self):
        """Run the interactive annotation tool"""
        if not self.image_files:
            print("❌ No images to annotate")
            return
        
        print(f"\n🎯 INTERACTIVE CHESS ANNOTATION TOOL")
        print(f"📁 Images: {len(self.image_files)}")
        print(f"♟️  Chess set: {self.chess_set}")
        print(f"📁 Output: {self.output_dir}")
        print(f"\n🎮 CONTROLS:")
        print(f"   • Click on image to place corners (TL, TR, BR, BL)")
        print(f"   • 'Auto' button: Use AI corner detection")
        print(f"   • 'Clear' button: Clear all corners")
        print(f"   • 'Accept' button: Save corners and enter FEN")
        print(f"   • 'Skip' button: Skip this image")
        print(f"   • 'Prev/Next' buttons: Navigate between images")
        print(f"   • 'Quit' button: Exit annotation tool")
        print(f"\n🚀 Starting annotation...")
        
        self.load_and_display_current()
        plt.show()

def main():
    """Main function"""
    image_dir = "/Users/tonyblum/Downloads"
    output_dir = "./chess_set2_annotations"
    chess_set = "set2"
    
    annotator = SimpleChessAnnotator(image_dir, output_dir, chess_set)
    annotator.run()

if __name__ == "__main__":
    main()
