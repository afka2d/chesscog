#!/usr/bin/env python3
"""
Interactive ground truth annotation tool for chess images.
This will help you create accurate ground truth data for evaluation.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveGroundTruthCreator:
    def __init__(self):
        self.current_image = None
        self.current_annotations = {}
        self.corners = []
        self.warped_board = None
        self.fig = None
        self.ax = None
        self.current_square = None
        self.square_rects = {}
        
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
    
    def create_square_grid(self, warped_board):
        """Create a grid overlay for the warped board"""
        board_size = warped_board.shape[0]
        square_size = board_size // 8
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.ax.imshow(warped_board)
        self.ax.set_title("Click on squares to annotate pieces")
        
        # Create square rectangles
        self.square_rects = {}
        for rank in range(8):
            for file in range(8):
                square_name = f"{chr(97+file)}{8-rank}"
                x1 = file * square_size
                y1 = rank * square_size
                
                rect = Rectangle((x1, y1), square_size, square_size, 
                               linewidth=2, edgecolor='yellow', facecolor='none')
                self.ax.add_patch(rect)
                self.square_rects[square_name] = rect
        
        # Add click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='green', label='Occupied - White'),
            mpatches.Patch(color='red', label='Occupied - Black'),
            mpatches.Patch(color='gray', label='Empty'),
            mpatches.Patch(color='yellow', label='Unannotated')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def on_click(self, event):
        """Handle mouse clicks on the board"""
        if event.inaxes != self.ax:
            return
        
        # Find which square was clicked
        board_size = 800
        square_size = board_size // 8
        
        file = int(event.xdata // square_size)
        rank = int(event.ydata // square_size)
        
        if 0 <= file < 8 and 0 <= rank < 8:
            square_name = f"{chr(97+file)}{8-rank}"
            self.current_square = square_name
            
            print(f"\nClicked on square: {square_name}")
            print("Options:")
            print("1. Empty")
            print("2. White Pawn")
            print("3. White Rook")
            print("4. White Knight")
            print("5. White Bishop")
            print("6. White Queen")
            print("7. White King")
            print("8. Black Pawn")
            print("9. Black Rook")
            print("10. Black Knight")
            print("11. Black Bishop")
            print("12. Black Queen")
            print("13. Black King")
            print("14. Skip this square")
            print("15. Done with this image")
            
            choice = input("Enter choice (1-15): ").strip()
            self.handle_choice(choice, square_name)
    
    def handle_choice(self, choice, square_name):
        """Handle the user's choice for a square"""
        piece_map = {
            '1': {'occupied': False, 'color': None, 'piece': None},
            '2': {'occupied': True, 'color': 'white', 'piece': 'pawn'},
            '3': {'occupied': True, 'color': 'white', 'piece': 'rook'},
            '4': {'occupied': True, 'color': 'white', 'piece': 'knight'},
            '5': {'occupied': True, 'color': 'white', 'piece': 'bishop'},
            '6': {'occupied': True, 'color': 'white', 'piece': 'queen'},
            '7': {'occupied': True, 'color': 'white', 'piece': 'king'},
            '8': {'occupied': True, 'color': 'black', 'piece': 'pawn'},
            '9': {'occupied': True, 'color': 'black', 'piece': 'rook'},
            '10': {'occupied': True, 'color': 'black', 'piece': 'knight'},
            '11': {'occupied': True, 'color': 'black', 'piece': 'bishop'},
            '12': {'occupied': True, 'color': 'black', 'piece': 'queen'},
            '13': {'occupied': True, 'color': 'black', 'piece': 'king'},
            '14': None,  # Skip
            '15': 'done'  # Done
        }
        
        if choice == '15':
            return 'done'
        elif choice == '14':
            return 'skip'
        elif choice in piece_map:
            annotation = piece_map[choice]
            if annotation is not None:
                self.current_annotations[square_name] = annotation
                
                # Update visual representation
                self.update_square_visual(square_name, annotation)
                
                print(f"Annotated {square_name}: {annotation}")
            return 'continue'
        else:
            print("Invalid choice. Please try again.")
            return 'continue'
    
    def update_square_visual(self, square_name, annotation):
        """Update the visual representation of a square"""
        if square_name in self.square_rects:
            rect = self.square_rects[square_name]
            
            if not annotation['occupied']:
                rect.set_facecolor('gray')
                rect.set_alpha(0.3)
            elif annotation['color'] == 'white':
                rect.set_facecolor('green')
                rect.set_alpha(0.3)
            elif annotation['color'] == 'black':
                rect.set_facecolor('red')
                rect.set_alpha(0.3)
            
            self.fig.canvas.draw()
    
    def create_annotation_for_image(self, image_path):
        """Create ground truth annotation for a single image"""
        logger.info(f"Creating annotation for: {Path(image_path).name}")
        
        # Load image
        img = cv2.imread(image_path)
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect corners
        corners = self.detect_chessboard_corners(image_path)
        self.corners = corners
        
        # Warp chessboard
        self.warped_board = self.warp_chessboard(img_array, corners)
        
        # Initialize annotations
        self.current_annotations = {}
        
        print(f"\nAnnotating image: {Path(image_path).name}")
        print("Click on squares to annotate pieces. Close the window when done.")
        
        # Create interactive grid
        self.create_square_grid(self.warped_board)
        
        return self.current_annotations
    
    def create_annotations_for_dataset(self, dataset_path, max_images=3):
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
            
            if annotation:
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
    creator = InteractiveGroundTruthCreator()
    
    # Create annotations for test/validation images
    dataset_path = "my_chess_images/train/images"  # Change to test/val when available
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    print("Interactive Chess Image Ground Truth Annotation Tool")
    print("=" * 60)
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
