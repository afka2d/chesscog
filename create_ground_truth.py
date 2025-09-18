#!/usr/bin/env python3
"""
Interactive tool to create ground truth annotations for chess images.
This will help you create accurate ground truth data for evaluation.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroundTruthCreator:
    def __init__(self):
        self.current_image = None
        self.current_annotations = {}
        self.corners = []
        self.warped_board = None
        self.current_square = None
        
    def detect_chessboard_corners(self, image_path):
        """Detect chessboard corners using OpenCV"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Convert to the format expected by the API
            corners_2d = corners.reshape(-1, 2)
            
            # Find the 4 outer corners
            top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
            top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
            bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
            bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
            
            return [top_left, top_right, bottom_right, bottom_left]
        else:
            # Fallback: estimate corners based on image dimensions
            h, w = img.shape[:2]
            margin = min(h, w) * 0.1
            
            return [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ]
    
    def warp_chessboard(self, img_array, corners_array):
        """Warp chessboard using the exact logic from the working commit."""
        corners = np.array(corners_array, dtype=np.float32)
        
        # Define destination points for a square board
        board_size = 800
        dst_points = np.array([
            [0, 0],
            [board_size - 1, 0],
            [board_size - 1, board_size - 1],
            [0, board_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        M = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img_array, M, (board_size, board_size))
        
        return warped
    
    def extract_square(self, warped_board, rank, file):
        """Extract a single square from the warped board."""
        board_size = warped_board.shape[0]
        square_size = board_size // 8
        
        x1 = file * square_size
        y1 = rank * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        return warped_board[y1:y2, x1:x2]
    
    def create_annotation_for_image(self, image_path):
        """Create ground truth annotation for a single image"""
        logger.info(f"Creating annotation for: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect corners
        corners = self.detect_chessboard_corners(image_path)
        self.corners = corners
        
        # Warp chessboard
        self.warped_board = self.warp_chessboard(img_array, corners)
        
        # Initialize annotations
        annotations = {}
        
        print(f"\nAnnotating image: {Path(image_path).name}")
        print("=" * 50)
        print("For each square, enter:")
        print("  'empty' or 'e' for empty squares")
        print("  'white_pawn', 'black_rook', etc. for pieces")
        print("  'skip' to skip this image")
        print("  'done' when finished")
        print()
        
        # Process each square
        for rank in range(8):
            for file in range(8):
                square_name = f"{chr(97+file)}{8-rank}"
                
                # Extract and display square
                square_img = self.extract_square(self.warped_board, rank, file)
                
                # Save square image for reference
                square_path = f"square_{square_name}.png"
                cv2.imwrite(square_path, cv2.cvtColor(square_img, cv2.COLOR_RGB2BGR))
                
                print(f"\nSquare {square_name}:")
                print(f"Square image saved as: {square_path}")
                
                while True:
                    user_input = input("Enter piece type (or 'empty', 'skip', 'done'): ").strip().lower()
                    
                    if user_input == 'done':
                        return annotations
                    elif user_input == 'skip':
                        return None
                    elif user_input in ['empty', 'e']:
                        annotations[square_name] = {
                            'occupied': False,
                            'color': None,
                            'piece': None
                        }
                        break
                    elif '_' in user_input and len(user_input.split('_')) == 2:
                        color, piece = user_input.split('_')
                        if color in ['white', 'black'] and piece in ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']:
                            annotations[square_name] = {
                                'occupied': True,
                                'color': color,
                                'piece': piece
                            }
                            break
                        else:
                            print("Invalid format. Use 'white_pawn', 'black_rook', etc.")
                    else:
                        print("Invalid input. Try again.")
                
                # Clean up square image
                Path(square_path).unlink()
        
        return annotations
    
    def create_annotations_for_dataset(self, dataset_path, max_images=5):
        """Create annotations for multiple images"""
        dataset_path = Path(dataset_path)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(images)} images")
        
        # Limit to max_images for now
        images = images[:max_images]
        
        all_annotations = {}
        
        for i, image_path in enumerate(images):
            print(f"\n{'='*60}")
            print(f"Image {i+1}/{len(images)}: {image_path.name}")
            print(f"{'='*60}")
            
            annotation = self.create_annotation_for_image(str(image_path))
            
            if annotation is not None:
                all_annotations[str(image_path)] = annotation
                
                # Save individual annotation file
                annotation_file = str(image_path).replace('.JPG', '.json').replace('.jpg', '.json')
                with open(annotation_file, 'w') as f:
                    json.dump(annotation, f, indent=2)
                
                print(f"Annotation saved to: {annotation_file}")
            else:
                print("Skipped this image")
        
        # Save combined annotations
        combined_file = "ground_truth_annotations.json"
        with open(combined_file, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        logger.info(f"All annotations saved to: {combined_file}")
        return all_annotations

def main():
    """Main function"""
    creator = GroundTruthCreator()
    
    # Create annotations for a few test images
    dataset_path = "my_chess_images/train/images"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    print("Chess Image Ground Truth Annotation Tool")
    print("=" * 50)
    print("This tool will help you create ground truth annotations")
    print("for evaluating model accuracy.")
    print()
    
    max_images = input("How many images to annotate? (default: 3): ").strip()
    if not max_images:
        max_images = 3
    else:
        max_images = int(max_images)
    
    creator.create_annotations_for_dataset(dataset_path, max_images)

if __name__ == "__main__":
    main()
