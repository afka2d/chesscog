#!/usr/bin/env python3
"""
Show Model Predictions with Images
Display actual images with model predictions for visual accuracy assessment
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import matplotlib.pyplot as plt
import random

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files may not load properly.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictionVisualizer:
    """Visualizer for model predictions with actual images"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Piece type mapping
        self.piece_types = {
            'P': 0, 'p': 0,  # pawn
            'N': 1, 'n': 1,  # knight
            'B': 2, 'b': 2,  # bishop
            'R': 3, 'r': 3,  # rook
            'Q': 4, 'q': 4,  # queen
            'K': 5, 'k': 5   # king
        }
        
        self.piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        self.num_classes = 6
        
    def load_model(self):
        """Load the combined piece classification model"""
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return None
        
        try:
            # Create model architecture (ResNet18)
            model = models.resnet18(weights=None)  # Don't load ImageNet weights
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 6)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"✅ Loaded combined piece classification model from: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def extract_piece_label_from_path(self, img_path):
        """Extract piece type label from image path"""
        try:
            # Try to extract from directory structure
            parts = str(img_path).split('/')
            for part in parts:
                # Handle grey_background_dataset format: black_pawn, white_king, etc.
                if any(piece in part.lower() for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']):
                    piece_name = part.lower()
                    if 'pawn' in piece_name:
                        return 0
                    elif 'knight' in piece_name:
                        return 1
                    elif 'bishop' in piece_name:
                        return 2
                    elif 'rook' in piece_name:
                        return 3
                    elif 'queen' in piece_name:
                        return 4
                    elif 'king' in piece_name:
                        return 5
            
            return None
        except:
            return None
    
    def get_transforms(self):
        """Get data transforms for validation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, model, image):
        """Predict piece type for a single image with confidence scores"""
        transform = self.get_transforms()
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, all_probs
    
    def load_grey_background_samples(self, num_samples=12):
        """Load grey background samples with images"""
        logger.info("Loading grey background samples...")
        
        # Load grey background data
        dataset = []
        data_paths = [
            "grey_background_dataset/pieces/train",
            "grey_background_dataset/pieces/val", 
            "grey_background_dataset/pieces/test"
        ]
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                # Load images and labels
                for img_file in path.rglob("*.png"):
                    try:
                        # Extract label from directory structure
                        label = self.extract_piece_label_from_path(img_file)
                        if label is not None:
                            # Load image
                            img = Image.open(img_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            dataset.append({
                                'image': img,
                                'label': label,
                                'path': str(img_file),
                                'filename': img_file.name
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if dataset:
                    break
        
        # Randomly sample
        if len(dataset) < num_samples:
            num_samples = len(dataset)
        
        samples = random.sample(dataset, num_samples)
        logger.info(f"Selected {len(samples)} grey background samples")
        return samples
    
    def load_marshall_samples(self, num_samples=12):
        """Load Marshall samples with images"""
        logger.info("Loading Marshall samples...")
        
        # Load Marshall annotations
        annotations_path = Path("marshall_chess_annotations/annotations.json")
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', {})
        excluded_images = set(data.get('excluded_images', []))
        
        # Filter out excluded images
        valid_annotations = {
            k: v for k, v in annotations.items() 
            if k not in excluded_images
        }
        
        # Create Marshall dataset
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        for image_name, annotation in valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
                continue
                
            try:
                # Load image (handle HEIC files)
                if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                    pil_image = Image.open(image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                else:
                    image = cv2.imread(str(image_path))
                
                if image is None:
                    continue
                    
                # Get corners and FEN
                corners = annotation.get('corners', [])
                fen = annotation.get('fen', '')
                
                if len(corners) != 4 or not fen:
                    continue
                
                # Warp board to get square images
                warped_board = self.warp_board(image, corners)
                if warped_board is None:
                    continue
                
                # Extract squares and create piece labels
                squares, piece_labels = self.extract_squares_with_piece_labels(warped_board, fen)
                
                for square, label in zip(squares, piece_labels):
                    if square is not None and label is not None:
                        # Convert BGR to RGB
                        if len(square.shape) == 3 and square.shape[2] == 3:
                            square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                        square_pil = Image.fromarray(square)
                        
                        dataset.append({
                            'image': square_pil,
                            'label': label,
                            'path': str(image_path),
                            'filename': f"{image_name}_{len(dataset)}"
                        })
                        
                        if len(dataset) >= num_samples * 5:  # Get more samples to choose from
                            break
                
                if len(dataset) >= num_samples * 5:
                    break
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
        
        # Randomly sample
        if len(dataset) < num_samples:
            num_samples = len(dataset)
        
        samples = random.sample(dataset, num_samples)
        logger.info(f"Selected {len(samples)} Marshall samples")
        return samples
    
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
    
    def extract_squares_with_piece_labels(self, warped_board, fen):
        """Extract 64 squares and create piece type labels from FEN"""
        squares = []
        labels = []
        
        # Parse FEN to get piece positions
        board = self.fen_to_board(fen)
        
        square_size = warped_board.shape[0] // 8
        
        for rank in range(8):
            for file in range(8):
                # Extract square
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped_board[y1:y2, x1:x2]
                squares.append(square)
                
                # Get piece at this position
                piece = board[rank][file]
                if piece == '.':
                    labels.append(None)  # Skip empty squares
                else:
                    label = self.piece_types.get(piece, None)
                    labels.append(label)
        
        return squares, labels
    
    def visualize_predictions(self, model, samples, dataset_name, save_path=None):
        """Visualize sample predictions with actual images"""
        logger.info(f"Creating visualization for {dataset_name}...")
        
        # Calculate grid size
        num_samples = len(samples)
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        correct = 0
        total = 0
        
        for idx, sample in enumerate(samples):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get prediction
            predicted_class, confidence, all_probs = self.predict_image(model, sample['image'])
            true_class = sample['label']
            
            is_correct = predicted_class == true_class
            if is_correct:
                correct += 1
            total += 1
            
            # Display image
            ax.imshow(sample['image'])
            ax.axis('off')
            
            # Create title with prediction info
            true_name = self.piece_names[true_class]
            pred_name = self.piece_names[predicted_class]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if is_correct else 'red'
            
            title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.3f}"
            ax.set_title(title, color=color, fontsize=12, fontweight='bold')
            
            # Add filename
            ax.text(0.02, 0.98, sample['filename'][:25], transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add top 3 predictions
            top3_indices = np.argsort(all_probs)[-3:][::-1]
            top3_probs = all_probs[top3_indices]
            top3_names = [self.piece_names[idx] for idx in top3_indices]
            
            top3_text = ', '.join([f'{name}({prob:.2f})' for name, prob in zip(top3_names, top3_probs)])
            ax.text(0.02, 0.02, f"Top 3: {top3_text}", transform=ax.transAxes, 
                   fontsize=7, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        # Add overall accuracy to the figure
        accuracy = 100.0 * correct / total
        fig.suptitle(f'{dataset_name} - Accuracy: {accuracy:.1f}% ({correct}/{total})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved: {save_path}")
        plt.close()
        
        return accuracy
    
    def run_visualization(self):
        """Run complete visualization"""
        logger.info("🚀 Starting Model Prediction Visualization")
        logger.info("=" * 60)
        
        # Load model
        model = self.load_model()
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Load samples
        grey_samples = self.load_grey_background_samples(num_samples=12)
        marshall_samples = self.load_marshall_samples(num_samples=12)
        
        # Visualize grey background samples
        logger.info("\n🔍 Visualizing Grey Background Samples...")
        grey_accuracy = self.visualize_predictions(
            model, grey_samples, "Grey Background Dataset", 
            "grey_background_predictions.png"
        )
        
        # Visualize Marshall samples
        logger.info("\n🔍 Visualizing Marshall Samples...")
        marshall_accuracy = self.visualize_predictions(
            model, marshall_samples, "Marshall Dataset", 
            "marshall_predictions.png"
        )
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 VISUALIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Grey Background Accuracy: {grey_accuracy:.1f}%")
        logger.info(f"Marshall Accuracy: {marshall_accuracy:.1f}%")
        logger.info(f"Average Accuracy: {(grey_accuracy + marshall_accuracy) / 2:.1f}%")
        logger.info("\n📁 Images saved:")
        logger.info("  - grey_background_predictions.png")
        logger.info("  - marshall_predictions.png")
        logger.info("\n✅ Visualization completed!")

def main():
    """Main visualization function"""
    visualizer = ModelPredictionVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
