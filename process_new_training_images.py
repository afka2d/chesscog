#!/usr/bin/env python3
"""
Enhanced script to process new training images with manual corner input and FEN verification.
This ensures high-quality training data while keeping the occupancy classifier untouched.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from PIL import Image
import logging
import chess
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessImageProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.annotations_dir = os.path.join(output_dir, "annotations")
        self.images_dir = os.path.join(output_dir, "images")
        
        # Create output directories
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Piece type mapping for FEN to folder names
        self.piece_mapping = {
            'P': 'white_pawn', 'R': 'white_rook', 'N': 'white_knight',
            'B': 'white_bishop', 'Q': 'white_queen', 'K': 'white_king',
            'p': 'black_pawn', 'r': 'black_rook', 'n': 'black_knight',
            'b': 'black_bishop', 'q': 'black_queen', 'k': 'black_king'
        }
    
    def get_image_files(self) -> List[str]:
        """Get list of image files to process."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        for file in os.listdir(self.input_dir):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_files.append(file)
        
        return sorted(image_files)
    
    def display_image_with_grid(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Display image with 8x8 grid overlay and allow manual corner input.
        Returns the image and corner coordinates.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Draw 8x8 grid
        grid_img = img_rgb.copy()
        
        # Vertical lines
        for i in range(9):
            x = int((width * i) / 8)
            cv2.line(grid_img, (x, 0), (x, height), (255, 0, 0), 2)
        
        # Horizontal lines
        for i in range(9):
            y = int((height * i) / 8)
            cv2.line(grid_img, (0, y), (width, y), (255, 0, 0), 2)
        
        # Add coordinate labels
        for rank in range(8):
            for file in range(8):
                x = int((width * file) / 8) + 20
                y = int((height * rank) / 8) + 30
                label = f"{chr(97+file)}{8-rank}"
                cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display image
        cv2.imshow('Chess Board with Grid - Click corners in order: a8, h8, h1, a1', grid_img)
        cv2.waitKey(1)
        
        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        print("🔍 Click the four corners in this order:")
        print("   1. Top-left (a8) - White's perspective")
        print("   2. Top-right (h8)")
        print("   3. Bottom-right (h1)")
        print("   4. Bottom-left (a1)")
        print("   Click each corner, then press 'Enter' to continue")
        
        # Get corner clicks
        corners = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                corners.append((x, y))
                # Draw circle at clicked point
                cv2.circle(grid_img, (x, y), 10, (0, 255, 0), -1)
                cv2.imshow('Chess Board with Grid - Click corners in order: a8, h8, h1, a1', grid_img)
                print(f"   Corner {len(corners)}: ({x}, {y})")
        
        cv2.setMouseCallback('Chess Board with Grid - Click corners in order: a8, h8, h1, a1', mouse_callback)
        
        # Wait for 4 corners
        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("User cancelled")
        
        cv2.destroyAllWindows()
        
        # Verify corners are in correct order
        if len(corners) == 4:
            print(f"✅ All corners captured: {corners}")
            return img, corners
        else:
            raise ValueError(f"Expected 4 corners, got {len(corners)}")
    
    def get_fen_input(self, image_name: str) -> str:
        """Get FEN input from user for the current position."""
        print(f"\n♟️  Enter the FEN for {image_name}:")
        print("   Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print("   (Press Enter to skip this image)")
        
        fen = input("FEN: ").strip()
        
        if not fen:
            return None
        
        # Validate FEN
        try:
            board = chess.Board(fen)
            print(f"✅ Valid FEN: {fen}")
            return fen
        except ValueError as e:
            print(f"❌ Invalid FEN: {e}")
            return self.get_fen_input(image_name)
    
    def extract_pieces_from_board(self, img: np.ndarray, corners: List[Tuple[int, int]], 
                                fen: str, image_name: str) -> int:
        """
        Extract individual pieces from the board using the provided corners and FEN.
        Returns the number of pieces extracted.
        """
        try:
            # Parse FEN to get piece positions
            board = chess.Board(fen)
            
            # Warp the chessboard to get a square grid
            warped = self.warp_chessboard(img, corners)
            
            # Calculate square size
            square_size = min(warped.shape[0], warped.shape[1]) // 8
            
            pieces_extracted = 0
            
            # Process each square
            for rank in range(8):
                for file in range(8):
                    square = chess.square(file, 7 - rank)  # Convert to chess coordinates
                    piece = board.piece_at(square)
                    
                    if piece is not None:
                        # Extract square image
                        x1 = file * square_size
                        y1 = rank * square_size
                        x2 = x1 + square_size
                        y2 = y1 + square_size
                        
                        square_img = warped[y1:y2, x1:x2]
                        
                        if square_img.size > 0:
                            # Resize to standard size
                            square_resized = cv2.resize(square_img, (100, 200))
                            
                            # Determine piece type and color
                            piece_char = piece.symbol()
                            folder_name = self.piece_mapping[piece_char]
                            
                            # Create folder if it doesn't exist
                            piece_folder = os.path.join(self.output_dir, "pieces", "train", folder_name)
                            os.makedirs(piece_folder, exist_ok=True)
                            
                            # Generate filename
                            piece_filename = f"{image_name}_{chr(97+file)}{8-rank}.png"
                            piece_path = os.path.join(piece_folder, piece_filename)
                            
                            # Save piece image
                            cv2.imwrite(piece_path, square_resized)
                            pieces_extracted += 1
            
            return pieces_extracted
            
        except Exception as e:
            logger.error(f"Error extracting pieces from {image_name}: {e}")
            return 0
    
    def warp_chessboard(self, img: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """Warp the chessboard to a square grid using the provided corners."""
        # Convert corners to numpy array
        src_corners = np.array(corners, dtype=np.float32)
        
        # Define destination corners (square grid)
        board_size = 400  # Size of warped board
        dst_corners = np.array([
            [0, 0],                    # Top-left
            [board_size, 0],           # Top-right
            [board_size, board_size],  # Bottom-right
            [0, board_size]            # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Apply transform
        warped = cv2.warpPerspective(img, transform_matrix, (board_size, board_size))
        
        return warped
    
    def save_annotation(self, image_name: str, corners: List[Tuple[int, int]], fen: str):
        """Save annotation data (corners and FEN) for the image."""
        annotation = {
            "image": image_name,
            "corners": corners,
            "fen": fen,
            "timestamp": str(Path(image_name).stem)
        }
        
        annotation_path = os.path.join(self.annotations_dir, f"{Path(image_name).stem}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"💾 Annotation saved: {annotation_path}")
    
    def process_images(self, start_from_image=303):
        """Main processing loop for all images."""
        image_files = self.get_image_files()
        
        if not image_files:
            print("❌ No image files found in input directory")
            return
        
        print(f"🚀 Found {len(image_files)} images to process")
        print(f"🎯 Starting from image {start_from_image}: {image_files[start_from_image-1] if start_from_image <= len(image_files) else 'END'}")
        print("=" * 60)
        
        total_pieces = 0
        processed_images = 0
        
        for i, image_file in enumerate(image_files, 1):
            # Skip images before the start point
            if i < start_from_image:
                continue
            try:
                print(f"\n📸 Processing image {i}/{len(image_files)}: {image_file}")
                
                image_path = os.path.join(self.input_dir, image_file)
                
                # Get corners manually
                img, corners = self.display_image_with_grid(image_path)
                
                # Get FEN input
                fen = self.get_fen_input(image_file)
                
                if fen is None:
                    print("⏭️  Skipping this image")
                    continue
                
                # Save annotation
                self.save_annotation(image_file, corners, fen)
                
                # Extract pieces
                pieces_extracted = self.extract_pieces_from_board(img, corners, fen, Path(image_file).stem)
                total_pieces += pieces_extracted
                processed_images += 1
                
                print(f"✅ Extracted {pieces_extracted} pieces from {image_file}")
                
                # Copy original image to output
                shutil.copy2(image_path, os.path.join(self.images_dir, image_file))
                
            except KeyboardInterrupt:
                print("\n⏹️  Processing interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
                continue
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print(f"📊 Images processed: {processed_images}/{len(image_files)}")
        print(f"♟️  Total pieces extracted: {total_pieces}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Pieces organized in: {self.output_dir}/pieces/train/")
        print(f"📁 Annotations saved in: {self.annotations_dir}")
        print(f"📁 Original images copied to: {self.images_dir}")
        
        if processed_images > 0:
            print(f"\n🚀 Your dataset has been enhanced by {total_pieces} new pieces!")
            print("💡 Next step: Retrain your piece classifier with the enhanced dataset")

def main():
    """Main function."""
    # Configuration
    input_dir = os.path.expanduser("~/Desktop/training_images_3")
    output_dir = "enhanced_training_dataset"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print("Please ensure your training images are in ~/Desktop/training_images_3/")
        return
    
    print("🎯 Enhanced Chess Training Image Processor")
    print("=" * 50)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print("\nThis script will:")
    print("1. Display each image with an 8x8 grid overlay")
    print("2. Allow you to click the four corners manually")
    print("3. Let you input the correct FEN for each position")
    print("4. Automatically extract individual pieces")
    print("5. Organize them into the correct training folders")
    print("\n⚠️  IMPORTANT: Your occupancy classifier will remain completely untouched!")
    
    input("\nPress Enter to continue...")
    
    # Create processor and start processing
    processor = ChessImageProcessor(input_dir, output_dir)
    processor.process_images(start_from_image=303)  # Start from image 303

if __name__ == "__main__":
    main()
