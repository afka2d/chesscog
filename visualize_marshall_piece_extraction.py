#!/usr/bin/env python3
"""
Visualize Marshall piece extraction process
Shows how board corners and FENs are used to generate individual piece photos
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files may not load properly.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarshallPieceVisualizer:
    def __init__(self, marshall_annotations_path="marshall_chess_annotations/annotations.json"):
        """Initialize the visualizer with Marshall data"""
        self.marshall_annotations_path = Path(marshall_annotations_path)
        self.marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        # Load Marshall annotations
        self.load_marshall_annotations()
        
    def load_marshall_annotations(self):
        """Load Marshall annotations"""
        logger.info("Loading Marshall annotations...")
        
        with open(self.marshall_annotations_path, 'r') as f:
            data = json.load(f)
        
        self.annotations = data.get('annotations', {})
        self.excluded_images = set(data.get('excluded_images', []))
        
        # Filter out excluded images
        self.valid_annotations = {
            k: v for k, v in self.annotations.items() 
            if k not in self.excluded_images
        }
        
        logger.info(f"Loaded {len(self.valid_annotations)} valid Marshall annotations")
    
    def fen_to_board(self, fen):
        """Convert FEN string to 8x8 board representation"""
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # Split FEN into parts
        parts = fen.split()
        if not parts:
            return board
        
        # Parse piece positions
        ranks = parts[0].split('/')
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    if file_idx < 8:
                        board[rank_idx][file_idx] = char
                        file_idx += 1
        
        return board
    
    def warp_board(self, image, corners):
        """Warp image to get a square chessboard"""
        try:
            # Convert corners to numpy array
            src_points = np.array(corners, dtype=np.float32)
            
            # Define destination points for a square board
            size = 400  # 400x400 pixel board
            dst_points = np.array([
                [0, 0],
                [size, 0],
                [size, size],
                [0, size]
            ], dtype=np.float32)
            
            # Get perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Warp image
            warped = cv2.warpPerspective(image, matrix, (size, size))
            
            return warped
        except Exception as e:
            logger.warning(f"Error warping board: {e}")
            return None
    
    def extract_squares(self, warped_board):
        """Extract 64 individual squares from warped board"""
        squares = []
        square_size = warped_board.shape[0] // 8
        
        for rank in range(8):
            for file in range(8):
                # Extract square
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped_board[y1:y2, x1:x2]
                squares.append((square, rank, file))
        
        return squares
    
    def visualize_piece_extraction(self, image_name, max_images=3):
        """Visualize the piece extraction process for a specific image"""
        if image_name not in self.valid_annotations:
            logger.error(f"Image {image_name} not found in valid annotations")
            return
        
        annotation = self.valid_annotations[image_name]
        image_path = self.marshall_photos_dir / image_name
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        
        try:
            # Load image
            if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return
            
            # Get corners and FEN
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                logger.error(f"Invalid annotation for {image_name}")
                return
            
            # Warp board
            warped_board = self.warp_board(image, corners)
            if warped_board is None:
                logger.error(f"Could not warp board for {image_name}")
                return
            
            # Extract squares
            squares = self.extract_squares(warped_board)
            
            # Parse FEN to get piece positions
            board = self.fen_to_board(fen)
            
            # Create visualization
            self.create_extraction_visualization(image, corners, warped_board, squares, board, image_name)
            
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
    
    def create_extraction_visualization(self, original_image, corners, warped_board, squares, board, image_name):
        """Create a comprehensive visualization of the piece extraction process"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original image with corners marked
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Original Image: {image_name}", fontsize=10)
        ax1.axis('off')
        
        # Mark corners
        for i, corner in enumerate(corners):
            ax1.plot(corner[0], corner[1], 'ro', markersize=8)
            ax1.text(corner[0], corner[1], f'{i+1}', color='red', fontsize=12, fontweight='bold')
        
        # 2. Warped board
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
        ax2.set_title("Warped Board (400x400)", fontsize=10)
        ax2.axis('off')
        
        # Add grid lines
        square_size = warped_board.shape[0] // 8
        for i in range(9):
            ax2.axhline(y=i * square_size, color='white', linewidth=1)
            ax2.axvline(x=i * square_size, color='white', linewidth=1)
        
        # 3. FEN board representation
        ax3 = plt.subplot(2, 4, 3)
        ax3.set_xlim(0, 8)
        ax3.set_ylim(0, 8)
        ax3.set_aspect('equal')
        ax3.set_title("FEN Board State", fontsize=10)
        ax3.set_xticks(range(9))
        ax3.set_yticks(range(9))
        ax3.grid(True, alpha=0.3)
        
        # Draw pieces
        for rank in range(8):
            for file in range(8):
                piece = board[rank][file]
                if piece != '.':
                    color = 'white' if piece.isupper() else 'black'
                    ax3.text(file + 0.5, 7 - rank + 0.5, piece, 
                            ha='center', va='center', fontsize=16, 
                            color=color, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray' if piece.isupper() else 'darkgray'))
        
        # 4-7. Sample individual squares
        sample_squares = [0, 7, 56, 63]  # Corners
        sample_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        for i, (square_idx, (rank, file)) in enumerate(zip(sample_squares, sample_positions)):
            ax = plt.subplot(2, 4, 4 + i)
            square_image, _, _ = squares[square_idx]
            ax.imshow(cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB))
            
            piece = board[rank][file]
            if piece != '.':
                color = 'White' if piece.isupper() else 'Black'
                ax.set_title(f"Square {chr(ord('a') + file)}{8 - rank}\n{color} {piece}", fontsize=10)
            else:
                ax.set_title(f"Square {chr(ord('a') + file)}{8 - rank}\nEmpty", fontsize=10)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'marshall_piece_extraction_{image_name.replace(".HEIC", "").replace(".heic", "")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print FEN for reference
        annotation = self.valid_annotations[image_name]
        print(f"\nFEN for {image_name}: {annotation.get('fen', 'N/A')}")
        print(f"Corners: {corners}")
    
    def visualize_multiple_images(self, max_images=3):
        """Visualize piece extraction for multiple images"""
        image_names = list(self.valid_annotations.keys())[:max_images]
        
        for image_name in image_names:
            logger.info(f"Visualizing piece extraction for: {image_name}")
            self.visualize_piece_extraction(image_name)
    
    def save_sample_pieces(self, image_name, output_dir="marshall_sample_pieces"):
        """Save individual piece images for manual inspection"""
        if image_name not in self.valid_annotations:
            logger.error(f"Image {image_name} not found in valid annotations")
            return
        
        annotation = self.valid_annotations[image_name]
        image_path = self.marshall_photos_dir / image_name
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Load and process image
            if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                logger.error(f"Invalid annotation for {image_name}")
                return
            
            # Warp board and extract squares
            warped_board = self.warp_board(image, corners)
            if warped_board is None:
                return
            
            squares = self.extract_squares(warped_board)
            board = self.fen_to_board(fen)
            
            # Save individual squares
            for rank in range(8):
                for file in range(8):
                    square_idx = rank * 8 + file
                    square_image, _, _ = squares[square_idx]
                    
                    piece = board[rank][file]
                    square_name = f"{chr(ord('a') + file)}{8 - rank}"
                    
                    if piece != '.':
                        color = 'white' if piece.isupper() else 'black'
                        filename = f"{image_name}_{square_name}_{color}_{piece}.jpg"
                    else:
                        filename = f"{image_name}_{square_name}_empty.jpg"
                    
                    cv2.imwrite(str(output_path / filename), square_image)
            
            logger.info(f"Saved individual pieces to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving pieces for {image_name}: {e}")

def main():
    """Main function to demonstrate piece extraction process"""
    logger.info("🔍 Marshall Piece Extraction Visualization")
    
    # Initialize visualizer
    visualizer = MarshallPieceVisualizer()
    
    if not visualizer.valid_annotations:
        logger.error("No valid Marshall annotations found")
        return
    
    # Show available images
    logger.info(f"Available images: {len(visualizer.valid_annotations)}")
    for i, image_name in enumerate(list(visualizer.valid_annotations.keys())[:5]):
        logger.info(f"  {i+1}. {image_name}")
    
    # Visualize first few images
    logger.info("\n📊 Creating visualizations...")
    visualizer.visualize_multiple_images(max_images=2)
    
    # Save sample pieces for manual inspection
    logger.info("\n💾 Saving sample pieces for manual inspection...")
    first_image = list(visualizer.valid_annotations.keys())[0]
    visualizer.save_sample_pieces(first_image)
    
    logger.info("\n✅ Visualization complete!")
    logger.info("Check the generated PNG files and the 'marshall_sample_pieces' directory for individual piece images.")

if __name__ == "__main__":
    main()
