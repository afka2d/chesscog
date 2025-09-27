#!/usr/bin/env python3
"""
Validate Combined Piece Classifier Model
Test accuracy on both grey background and Marshall datasets
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class PieceDataset(Dataset):
    """Dataset for piece classification validation"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        square = item['square']
        label = item['label']
        
        # Handle different data formats
        if isinstance(square, torch.Tensor):
            # Already a tensor (from previous data)
            square_pil = transforms.ToPILImage()(square)
        else:
            # BGR image (from Marshall data)
            if len(square.shape) == 3 and square.shape[2] == 3:
                square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square_pil = Image.fromarray(square)
        
        if self.transform:
            square_pil = self.transform(square_pil)
        
        return square_pil, label

class CombinedPieceValidator:
    """Validator for combined piece classification model"""
    
    def __init__(self, marshall_annotations_path="marshall_chess_annotations/annotations.json"):
        """Initialize the validator"""
        self.marshall_annotations_path = Path(marshall_annotations_path)
        self.model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
        
        # Load Marshall annotations
        self.load_marshall_annotations()
        
        # Model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Piece type mapping (6 classes: pawn, knight, bishop, rook, queen, king)
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
    
    def create_marshall_dataset(self):
        """Create Marshall piece classification dataset"""
        logger.info("Creating Marshall piece classification dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        processed = 0
        errors = 0
        
        for image_name, annotation in self.valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
                errors += 1
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
                        dataset.append({
                            'square': square,
                            'label': label,
                            'source': 'marshall',
                            'image_name': image_name
                        })
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"Processed {processed} Marshall images, created {len(dataset)} samples")
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                errors += 1
                continue
        
        logger.info(f"Created Marshall dataset with {len(dataset)} samples")
        logger.info(f"Processed {processed} Marshall images, {errors} errors")
        return dataset
    
    def load_grey_background_data(self):
        """Load grey background dataset"""
        logger.info("Loading grey background dataset...")
        
        dataset = []
        data_paths = [
            "grey_background_dataset/pieces/train",
            "grey_background_dataset/pieces/val", 
            "grey_background_dataset/pieces/test"
        ]
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                logger.info(f"Found grey background data at: {path}")
                
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
                            
                            # Convert to tensor
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(img)
                            
                            dataset.append({
                                'square': img_tensor,
                                'label': label,
                                'source': 'grey_background',
                                'image_name': img_file.name
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if dataset:
                    break
        
        logger.info(f"Loaded {len(dataset)} grey background training samples")
        return dataset
    
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
    
    def evaluate_model(self, model, dataset, dataset_name):
        """Evaluate model on a dataset"""
        logger.info(f"\n🔍 Evaluating combined model on {dataset_name}...")
        
        if len(dataset) == 0:
            logger.warning(f"No data available for {dataset_name}")
            return None
        
        # Create data loader
        val_dataset = PieceDataset(dataset, self.get_transforms())
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Evaluate
        all_predictions = []
        all_labels = []
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Calculate overall accuracy
        correct = sum(class_correct)
        total = sum(class_total)
        overall_acc = 100.0 * correct / total
        
        logger.info(f"📊 {dataset_name} Results:")
        logger.info(f"Overall accuracy: {overall_acc:.2f}% ({correct}/{total})")
        
        # Per-class accuracy
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"{self.piece_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Classification report
        logger.info(f"\n📋 Detailed Classification Report for {dataset_name}:")
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.piece_names, 
                                     digits=3)
        logger.info(report)
        
        return {
            'overall_accuracy': overall_acc,
            'class_accuracies': [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                               for i in range(self.num_classes)],
            'class_totals': class_total,
            'class_correct': class_correct,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def create_confusion_matrix(self, results, dataset_name):
        """Create and save confusion matrix"""
        if results is None:
            return
        
        # Create confusion matrix
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.piece_names, 
                   yticklabels=self.piece_names)
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        output_path = f"confusion_matrix_combined_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Confusion matrix saved: {output_path}")
        plt.close()
    
    def run_validation(self):
        """Run complete validation"""
        logger.info("🚀 Starting Combined Piece Classifier Model Validation")
        logger.info("=" * 70)
        
        # Load model
        model = self.load_model()
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Create datasets
        marshall_data = self.create_marshall_dataset()
        grey_data = self.load_grey_background_data()
        
        # Evaluate on Marshall data
        marshall_results = self.evaluate_model(model, marshall_data, "Marshall Data")
        if marshall_results:
            self.create_confusion_matrix(marshall_results, "Marshall Data")
        
        # Evaluate on grey background data
        grey_results = self.evaluate_model(model, grey_data, "Grey Background Data")
        if grey_results:
            self.create_confusion_matrix(grey_results, "Grey Background Data")
        
        # Summary comparison
        logger.info("\n" + "=" * 70)
        logger.info("📊 SUMMARY COMPARISON")
        logger.info("=" * 70)
        
        if marshall_results:
            logger.info(f"Marshall Data Accuracy: {marshall_results['overall_accuracy']:.2f}%")
        else:
            logger.info("Marshall Data Accuracy: N/A (no data)")
        
        if grey_results:
            logger.info(f"Grey Background Data Accuracy: {grey_results['overall_accuracy']:.2f}%")
        else:
            logger.info("Grey Background Data Accuracy: N/A (no data)")
        
        # Per-class comparison
        if marshall_results and grey_results:
            logger.info("\n📋 Per-Class Accuracy Comparison:")
            logger.info("Piece Type | Marshall | Grey Background | Difference")
            logger.info("-" * 60)
            for i, piece_name in enumerate(self.piece_names):
                marshall_acc = marshall_results['class_accuracies'][i]
                grey_acc = grey_results['class_accuracies'][i]
                diff = marshall_acc - grey_acc
                logger.info(f"{piece_name:10} | {marshall_acc:8.2f}% | {grey_acc:13.2f}% | {diff:+7.2f}%")
        
        logger.info("\n✅ Validation completed!")

def main():
    """Main validation function"""
    validator = CombinedPieceValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
