#!/usr/bin/env python3
"""
Semi-Automated Chess Annotation Tool
===================================

For annotating new chess set images with corners and FEN positions.
Uses existing corner detection to speed up the process while allowing manual corrections.

Features:
- Auto-detects corners using robust API (Port 8005)
- Interactive corner adjustment with visual feedback
- FEN position input with validation
- Chess set labeling (Set 1 vs Set 2)
- Batch processing capabilities
- Quality control and validation
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import requests
import chess
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiAutomatedAnnotationTool:
    """
    Interactive tool for annotating chess images with corners and FEN positions
    """
    
    def __init__(self, images_dir: str, output_dir: str, chess_set: str = "set2"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.chess_set = chess_set
        self.corner_api_url = "http://localhost:8005"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Annotation state
        self.current_image = None
        self.current_corners = None
        self.current_fen = None
        self.image_files = []
        self.current_index = 0
        
        # Load image files
        self._load_image_files()
        
        logger.info(f"📁 Loaded {len(self.image_files)} images from {self.images_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"♟️  Chess set: {self.chess_set}")
    
    def _load_image_files(self):
        """Load all image files from the directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for ext in image_extensions:
            self.image_files.extend(list(self.images_dir.glob(f"*{ext}")))
        
        self.image_files.sort()
        logger.info(f"Found {len(self.image_files)} image files")
    
    def _get_auto_corners(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Get automatic corner detection from robust API
        """
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f"{self.corner_api_url}/detect_corners",
                    files={'file': f},
                    params={'time_budget': 2.0},
                    timeout=10
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    corners = data['corners']
                    logger.info(f"✅ Auto-detected corners: {corners}")
                    return corners
                else:
                    logger.warning("❌ Auto-detection failed")
                    return None
            else:
                logger.warning(f"❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Auto-detection error: {e}")
            return None
    
    def _validate_fen(self, fen: str) -> bool:
        """
        Validate FEN string
        """
        try:
            board = chess.Board(fen)
            return True
        except:
            return False
    
    def _draw_corners_on_image(self, image: np.ndarray, corners: List[List[float]], 
                              title: str = "Chess Board Corners") -> np.ndarray:
        """
        Draw corners on image for visualization
        """
        vis_img = image.copy()
        corners_np = np.array(corners, dtype=np.int32)
        
        # Draw corners
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
            x, y = corner
            cv2.circle(vis_img, (x, y), 15, color, -1)
            cv2.circle(vis_img, (x, y), 20, (255, 255, 255), 3)
            cv2.putText(vis_img, f'{label}', (x-20, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw quadrilateral
        cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        
        # Add title
        cv2.putText(vis_img, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(vis_img, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        return vis_img
    
    def _interactive_corner_adjustment(self, image: np.ndarray, 
                                     initial_corners: List[List[float]]) -> List[List[float]]:
        """
        Interactive corner adjustment using mouse clicks
        """
        corners = initial_corners.copy()
        current_corner = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_corner, corners
            
            if event == cv2.EVENT_LBUTTONDOWN:
                corners[current_corner] = [float(x), float(y)]
                current_corner = (current_corner + 1) % 4
                
                # Redraw image
                vis_img = self._draw_corners_on_image(image, corners, 
                    f"Adjusting corner {current_corner + 1}/4 (Click to adjust)")
                cv2.imshow("Corner Adjustment", vis_img)
        
        # Create window and set mouse callback
        cv2.namedWindow("Corner Adjustment", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Corner Adjustment", mouse_callback)
        
        # Initial display
        vis_img = self._draw_corners_on_image(image, corners, 
            f"Adjusting corner {current_corner + 1}/4 (Click to adjust)")
        cv2.imshow("Corner Adjustment", vis_img)
        
        print("\n🎯 CORNER ADJUSTMENT MODE")
        print("=" * 50)
        print("Instructions:")
        print("• Click on each corner to adjust its position")
        print("• Corners will be adjusted in order: TL → TR → BR → BL")
        print("• Press 's' to save and continue")
        print("• Press 'r' to reset to auto-detected corners")
        print("• Press 'q' to quit without saving")
        print("• Press 'n' to skip this image")
        print("=" * 50)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                print("✅ Corners saved!")
                break
            elif key == ord('r'):
                corners = initial_corners.copy()
                current_corner = 0
                vis_img = self._draw_corners_on_image(image, corners, 
                    f"Reset to auto-detected corners. Adjusting corner {current_corner + 1}/4")
                cv2.imshow("Corner Adjustment", vis_img)
                print("🔄 Reset to auto-detected corners")
            elif key == ord('q'):
                print("❌ Quitting without saving...")
                cv2.destroyAllWindows()
                return None
            elif key == ord('n'):
                print("⏭️  Skipping this image...")
                cv2.destroyAllWindows()
                return "skip"
        
        cv2.destroyAllWindows()
        return corners
    
    def _input_fen_position(self) -> Optional[str]:
        """
        Interactive FEN position input with validation
        """
        print("\n♟️  FEN POSITION INPUT")
        print("=" * 50)
        print("Enter the FEN position for this chess board.")
        print("Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print("(You can omit the last part: 'w KQkq - 0 1' if you want)")
        print("=" * 50)
        
        while True:
            fen_input = input("FEN: ").strip()
            
            if not fen_input:
                print("❌ Empty FEN. Please enter a valid FEN or 'skip' to skip this image.")
                continue
            
            if fen_input.lower() == 'skip':
                return "skip"
            
            # Add default ending if not provided
            if ' ' not in fen_input:
                fen_input += " w KQkq - 0 1"
            
            # Validate FEN
            if self._validate_fen(fen_input):
                print(f"✅ Valid FEN: {fen_input}")
                return fen_input
            else:
                print("❌ Invalid FEN. Please check the format and try again.")
                print("   Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    def _save_annotation(self, image_path: Path, corners: List[List[float]], 
                        fen: str, chess_set: str) -> bool:
        """
        Save annotation to JSON file
        """
        try:
            annotation = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "chess_set": chess_set,
                "corners": corners,
                "fen": fen,
                "annotation_method": "semi_automated",
                "corner_detection_api": "robust_port_8005",
                "timestamp": str(Path.cwd())
            }
            
            # Save annotation
            annotation_file = self.output_dir / "annotations" / f"{image_path.stem}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            # Save visualization
            image = cv2.imread(str(image_path))
            if image is not None:
                vis_img = self._draw_corners_on_image(image, corners, 
                    f"Set {chess_set} - {image_path.name}")
                vis_file = self.output_dir / "visualizations" / f"{image_path.stem}_corners.jpg"
                cv2.imwrite(str(vis_file), vis_img)
            
            logger.info(f"✅ Saved annotation: {annotation_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save annotation: {e}")
            return False
    
    def annotate_single_image(self, image_path: Path) -> bool:
        """
        Annotate a single image
        """
        print(f"\n📸 ANNOTATING: {image_path.name}")
        print("=" * 60)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Step 1: Auto-detect corners
        print("🔍 Step 1: Auto-detecting corners...")
        auto_corners = self._get_auto_corners(str(image_path))
        
        if auto_corners is None:
            print("❌ Auto-detection failed. Please enter corners manually.")
            print("Enter 4 corners as: x1,y1 x2,y2 x3,y3 x4,y4")
            corners_input = input("Corners: ").strip()
            
            if corners_input.lower() == 'skip':
                return False
            
            try:
                # Parse manual input
                corner_pairs = corners_input.split()
                corners = []
                for pair in corner_pairs:
                    x, y = map(float, pair.split(','))
                    corners.append([x, y])
                
                if len(corners) != 4:
                    print("❌ Must provide exactly 4 corners")
                    return False
                    
            except Exception as e:
                print(f"❌ Invalid corner format: {e}")
                return False
        else:
            corners = auto_corners
        
        # Step 2: Interactive corner adjustment
        print("🎯 Step 2: Interactive corner adjustment...")
        adjusted_corners = self._interactive_corner_adjustment(image, corners)
        
        if adjusted_corners is None:
            print("❌ Annotation cancelled")
            return False
        elif adjusted_corners == "skip":
            print("⏭️  Image skipped")
            return False
        
        # Step 3: FEN position input
        print("♟️  Step 3: FEN position input...")
        fen = self._input_fen_position()
        
        if fen is None:
            print("❌ Annotation cancelled")
            return False
        elif fen == "skip":
            print("⏭️  Image skipped")
            return False
        
        # Step 4: Save annotation
        print("💾 Step 4: Saving annotation...")
        success = self._save_annotation(image_path, adjusted_corners, fen, self.chess_set)
        
        if success:
            print(f"✅ Successfully annotated {image_path.name}")
            return True
        else:
            print(f"❌ Failed to save annotation for {image_path.name}")
            return False
    
    def annotate_all_images(self, start_index: int = 0):
        """
        Annotate all images starting from given index
        """
        total_images = len(self.image_files)
        successful = 0
        skipped = 0
        
        print(f"\n🚀 STARTING BATCH ANNOTATION")
        print("=" * 60)
        print(f"📁 Total images: {total_images}")
        print(f"🎯 Starting from index: {start_index}")
        print(f"♟️  Chess set: {self.chess_set}")
        print("=" * 60)
        
        for i in range(start_index, total_images):
            image_path = self.image_files[i]
            
            print(f"\n📊 Progress: {i + 1}/{total_images} ({((i + 1) / total_images) * 100:.1f}%)")
            
            try:
                success = self.annotate_single_image(image_path)
                
                if success:
                    successful += 1
                else:
                    skipped += 1
                
                # Ask if user wants to continue
                if i < total_images - 1:
                    continue_choice = input(f"\nContinue to next image? (y/n/q): ").strip().lower()
                    if continue_choice == 'n':
                        print("⏸️  Paused. You can resume later with the same start index.")
                        break
                    elif continue_choice == 'q':
                        print("🛑 Stopping annotation process.")
                        break
                        
            except KeyboardInterrupt:
                print(f"\n⏸️  Interrupted at image {i + 1}. You can resume with start_index={i}")
                break
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                skipped += 1
        
        # Summary
        print(f"\n📊 ANNOTATION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully annotated: {successful}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"📁 Total processed: {successful + skipped}")
        print(f"📁 Remaining: {total_images - (successful + skipped)}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def resume_annotation(self, start_index: int):
        """
        Resume annotation from a specific index
        """
        print(f"🔄 Resuming annotation from index {start_index}")
        self.annotate_all_images(start_index)
    
    def validate_annotations(self):
        """
        Validate all created annotations
        """
        print("\n🔍 VALIDATING ANNOTATIONS")
        print("=" * 50)
        
        annotation_dir = self.output_dir / "annotations"
        annotation_files = list(annotation_dir.glob("*.json"))
        
        valid_count = 0
        invalid_count = 0
        
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Check required fields
                required_fields = ['image_path', 'corners', 'fen', 'chess_set']
                if all(field in data for field in required_fields):
                    # Validate FEN
                    if self._validate_fen(data['fen']):
                        # Validate corners
                        if len(data['corners']) == 4 and all(len(corner) == 2 for corner in data['corners']):
                            valid_count += 1
                        else:
                            print(f"❌ Invalid corners in {ann_file.name}")
                            invalid_count += 1
                    else:
                        print(f"❌ Invalid FEN in {ann_file.name}")
                        invalid_count += 1
                else:
                    print(f"❌ Missing fields in {ann_file.name}")
                    invalid_count += 1
                    
            except Exception as e:
                print(f"❌ Error reading {ann_file.name}: {e}")
                invalid_count += 1
        
        print(f"\n📊 VALIDATION RESULTS")
        print(f"✅ Valid annotations: {valid_count}")
        print(f"❌ Invalid annotations: {invalid_count}")
        print(f"📁 Total files: {len(annotation_files)}")

def main():
    """
    Main function for the annotation tool
    """
    print("🎯 SEMI-AUTOMATED CHESS ANNOTATION TOOL")
    print("=" * 60)
    print("This tool helps you annotate chess images with corners and FEN positions.")
    print("It uses the robust corner detection API to speed up the process.")
    print("=" * 60)
    
    # Get input directory
    images_dir = input("📁 Enter path to images directory: ").strip()
    if not Path(images_dir).exists():
        print(f"❌ Directory not found: {images_dir}")
        return
    
    # Get output directory
    output_dir = input("📁 Enter output directory (default: ./chess_set2_annotations): ").strip()
    if not output_dir:
        output_dir = "./chess_set2_annotations"
    
    # Get chess set
    chess_set = input("♟️  Enter chess set name (default: set2): ").strip()
    if not chess_set:
        chess_set = "set2"
    
    # Create tool
    tool = SemiAutomatedAnnotationTool(images_dir, output_dir, chess_set)
    
    # Check if robust API is available
    try:
        response = requests.get(f"{tool.corner_api_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Robust corner detection API not available. Please start it first:")
            print("   python robust_corner_api.py")
            return
    except:
        print("❌ Robust corner detection API not available. Please start it first:")
        print("   python robust_corner_api.py")
        return
    
    print("✅ Robust corner detection API is available")
    
    # Main menu
    while True:
        print(f"\n📋 MAIN MENU")
        print("=" * 30)
        print("1. Annotate all images")
        print("2. Annotate single image")
        print("3. Resume from specific index")
        print("4. Validate annotations")
        print("5. Exit")
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == '1':
            start_idx = input("Start from index (default: 0): ").strip()
            start_idx = int(start_idx) if start_idx.isdigit() else 0
            tool.annotate_all_images(start_idx)
            
        elif choice == '2':
            if tool.image_files:
                print(f"\nAvailable images:")
                for i, img in enumerate(tool.image_files):
                    print(f"  {i}: {img.name}")
                
                img_idx = input("Enter image index: ").strip()
                if img_idx.isdigit() and 0 <= int(img_idx) < len(tool.image_files):
                    tool.annotate_single_image(tool.image_files[int(img_idx)])
                else:
                    print("❌ Invalid image index")
            else:
                print("❌ No images found")
                
        elif choice == '3':
            start_idx = input("Resume from index: ").strip()
            if start_idx.isdigit():
                tool.resume_annotation(int(start_idx))
            else:
                print("❌ Invalid index")
                
        elif choice == '4':
            tool.validate_annotations()
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
