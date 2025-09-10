#!/usr/bin/env python3
"""
Comprehensive evaluation of piece classifier models on real-world accuracy.
Tests each model on actual chess images and measures accuracy by piece type.
"""

import logging
import json
import numpy as np
import cv2
import torch
import chess
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PieceClassifierEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.test_images = []
        self.ground_truth = []
        
        # Piece class mapping
        self.piece_classes = [
            'white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king',
            'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king'
        ]
        
        # Load test images and ground truth
        self.load_test_data()
        
    def load_test_data(self):
        """Load test images and ground truth data"""
        logger.info("Loading test data...")
        
        # Load test images from the dataset
        test_dir = Path("grey_background_dataset/pieces/test")
        if not test_dir.exists():
            logger.error("Test dataset not found. Please ensure grey_background_dataset exists.")
            return
            
        for piece_class in self.piece_classes:
            piece_dir = test_dir / piece_class
            if piece_dir.exists():
                for img_path in piece_dir.glob("*.png"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            self.test_images.append({
                                'path': str(img_path),
                                'piece_class': piece_class,
                                'image': img
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
        
        logger.info(f"Loaded {len(self.test_images)} test images")
        
        # Create ground truth labels
        for item in self.test_images:
            piece_class = item['piece_class']
            if 'white' in piece_class:
                color = chess.WHITE
            else:
                color = chess.BLACK
                
            piece_type = piece_class.split('_')[1]
            piece_type_map = {
                'pawn': chess.PAWN,
                'rook': chess.ROOK,
                'knight': chess.KNIGHT,
                'bishop': chess.BISHOP,
                'queen': chess.QUEEN,
                'king': chess.KING
            }
            
            piece = chess.Piece(piece_type_map[piece_type], color)
            self.ground_truth.append(piece)
    
    def load_models(self):
        """Load all available piece classifier models"""
        logger.info("Loading piece classifier models...")
        
        model_paths = [
            ("ResNet_lightweight", "runs/piece_classifier/ResNet_lightweight/ResNet_lightweight.pt"),
            ("ResNet_uniform", "runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt"),
            ("ResNet_robust", "models/piece_classifier/ResNet_robust.pt"),
            ("ResNet_robust_full", "models/piece_classifier/ResNet_robust_full.pt"),
            ("ResNet_simple", "models/piece_classifier/ResNet_simple.pt"),
            ("ResNet_simple_balanced", "models/piece_classifier/ResNet_simple_balanced.pt"),
            ("ResNet_simple_robust", "models/piece_classifier/ResNet_simple_robust.pt"),
            ("InceptionV3", "models/piece_classifier/InceptionV3.pt"),
            ("TwoStage", "two_stage_models/piece_type_classifier.pt"),
        ]
        
        for model_name, model_path in model_paths:
            try:
                if Path(model_path).exists():
                    logger.info(f"Loading {model_name}...")
                    model = torch.load(str(model_path), map_location='cpu', weights_only=False)
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"✓ {model_name} loaded successfully")
                else:
                    logger.warning(f"Model not found: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    def warp_chessboard(self, img, corners):
        """Warp chessboard image to 800x800 with 100x100 squares"""
        # Define target points for 800x800 image
        target_size = 800
        square_size = 100
        
        # Create target corners for 800x800 image
        target_corners = np.array([
            [0, 0],
            [target_size - 1, 0],
            [target_size - 1, target_size - 1],
            [0, target_size - 1]
        ], dtype=np.float32)
        
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
        
        # Warp the image
        warped = cv2.warpPerspective(img, M, (target_size, target_size))
        
        return warped
    
    def extract_square(self, warped_img, rank, file, square_size=100):
        """Extract a square from the warped chessboard"""
        y_start = rank * square_size
        y_end = y_start + square_size
        x_start = file * square_size
        x_end = x_start + square_size
        
        square = warped_img[y_start:y_end, x_start:x_end]
        return square
    
    def preprocess_square(self, square_img):
        """Preprocess square image for model input"""
        # Resize to 224x224 (standard for ResNet)
        square_resized = cv2.resize(square_img, (224, 224))
        
        # Convert BGR to RGB
        square_rgb = cv2.cvtColor(square_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(square_rgb)
        
        # Apply transforms (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(pil_img)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_piece(self, model, square_img):
        """Predict piece type from square image"""
        with torch.no_grad():
            input_tensor = self.preprocess_square(square_img)
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model on all test images"""
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            'total_correct': 0,
            'total_predictions': 0,
            'piece_accuracy': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'piece_predictions': defaultdict(int),
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'predictions': []
        }
        
        # Create dummy corners for warping (we'll use the center of each image)
        for i, (test_item, ground_truth_piece) in enumerate(zip(self.test_images, self.ground_truth)):
            try:
                img = test_item['image']
                h, w = img.shape[:2]
                
                # Create dummy corners (center of image with some margin)
                margin = min(w, h) // 4
                corners = np.array([
                    [margin, margin],
                    [w - margin, margin],
                    [w - margin, h - margin],
                    [margin, h - margin]
                ], dtype=np.float32)
                
                # Warp the image
                warped = self.warp_chessboard(img, corners)
                
                # Extract center square (rank 4, file 4)
                square = self.extract_square(warped, 4, 4)
                
                # Predict piece
                predicted_class, confidence = self.predict_piece(model, square)
                
                # Convert predicted class to piece
                predicted_piece = self.class_to_piece(predicted_class)
                
                # Check if prediction is correct
                is_correct = predicted_piece == ground_truth_piece
                
                # Update results
                results['total_predictions'] += 1
                if is_correct:
                    results['total_correct'] += 1
                
                # Update piece-specific accuracy
                piece_name = self.piece_to_name(ground_truth_piece)
                results['piece_accuracy'][piece_name]['total'] += 1
                if is_correct:
                    results['piece_accuracy'][piece_name]['correct'] += 1
                
                # Update piece predictions count
                predicted_piece_name = self.piece_to_name(predicted_piece)
                results['piece_predictions'][predicted_piece_name] += 1
                
                # Update confusion matrix
                results['confusion_matrix'][piece_name][predicted_piece_name] += 1
                
                # Store prediction details
                results['predictions'].append({
                    'image_path': test_item['path'],
                    'ground_truth': piece_name,
                    'predicted': predicted_piece_name,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.test_images)} images")
                    
            except Exception as e:
                logger.warning(f"Failed to process {test_item['path']}: {e}")
                continue
        
        # Calculate overall accuracy
        results['overall_accuracy'] = results['total_correct'] / results['total_predictions'] if results['total_predictions'] > 0 else 0
        
        # Calculate piece-specific accuracy
        for piece_name in results['piece_accuracy']:
            piece_data = results['piece_accuracy'][piece_name]
            piece_data['accuracy'] = piece_data['correct'] / piece_data['total'] if piece_data['total'] > 0 else 0
        
        return results
    
    def class_to_piece(self, class_idx):
        """Convert class index to chess piece"""
        piece_class = self.piece_classes[class_idx]
        
        if 'white' in piece_class:
            color = chess.WHITE
        else:
            color = chess.BLACK
            
        piece_type = piece_class.split('_')[1]
        piece_type_map = {
            'pawn': chess.PAWN,
            'rook': chess.ROOK,
            'knight': chess.KNIGHT,
            'bishop': chess.BISHOP,
            'queen': chess.QUEEN,
            'king': chess.KING
        }
        
        return chess.Piece(piece_type_map[piece_type], color)
    
    def piece_to_name(self, piece):
        """Convert chess piece to string name"""
        if piece is None:
            return "empty"
            
        color = "white" if piece.color == chess.WHITE else "black"
        piece_type = piece.symbol().lower() if piece.color == chess.WHITE else piece.symbol().upper()
        
        piece_name_map = {
            'p': 'pawn', 'P': 'pawn',
            'r': 'rook', 'R': 'rook',
            'n': 'knight', 'N': 'knight',
            'b': 'bishop', 'B': 'bishop',
            'q': 'queen', 'Q': 'queen',
            'k': 'king', 'K': 'king'
        }
        
        return f"{color}_{piece_name_map[piece_type]}"
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        logger.info("Starting evaluation of all models...")
        
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {model_name}")
            logger.info(f"{'='*50}")
            
            start_time = time.time()
            results = self.evaluate_model(model_name, model)
            end_time = time.time()
            
            results['evaluation_time'] = end_time - start_time
            self.results[model_name] = results
            
            # Print results
            self.print_model_results(model_name, results)
    
    def print_model_results(self, model_name, results):
        """Print detailed results for a model"""
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['total_correct']}/{results['total_predictions']})")
        print(f"Evaluation Time: {results['evaluation_time']:.2f} seconds")
        
        print(f"\nPiece-Specific Accuracy:")
        print(f"{'Piece':<15} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
        print(f"{'-'*45}")
        
        for piece_name in sorted(results['piece_accuracy'].keys()):
            piece_data = results['piece_accuracy'][piece_name]
            accuracy = piece_data['accuracy']
            correct = piece_data['correct']
            total = piece_data['total']
            print(f"{piece_name:<15} {accuracy:<10.3f} {correct:<8} {total:<8}")
        
        print(f"\nPiece Prediction Distribution:")
        print(f"{'Piece':<15} {'Count':<8} {'Percentage':<10}")
        print(f"{'-'*35}")
        
        total_predictions = results['total_predictions']
        for piece_name, count in sorted(results['piece_predictions'].items()):
            percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"{piece_name:<15} {count:<8} {percentage:<10.1f}%")
        
        # Check for diversity (avoid models that predict only one piece type)
        unique_predictions = len(set(results['piece_predictions'].keys()))
        print(f"\nDiversity Score: {unique_predictions}/12 unique piece types predicted")
        
        if unique_predictions < 6:
            print("⚠️  WARNING: Low diversity - model may be overfitting to specific piece types")
    
    def find_best_model(self):
        """Find the best model based on overall accuracy and diversity"""
        logger.info("\nFinding best model...")
        
        best_model = None
        best_score = -1
        
        for model_name, results in self.results.items():
            # Calculate composite score: 70% accuracy + 30% diversity
            accuracy = results['overall_accuracy']
            diversity = len(set(results['piece_predictions'].keys())) / 12  # Normalize to 0-1
            
            composite_score = 0.7 * accuracy + 0.3 * diversity
            
            print(f"{model_name}: Accuracy={accuracy:.3f}, Diversity={diversity:.3f}, Score={composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_name
        
        print(f"\n🏆 BEST MODEL: {best_model} (Score: {best_score:.3f})")
        return best_model, best_score
    
    def save_results(self):
        """Save detailed results to file"""
        results_file = "piece_classifier_evaluation_results.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                'overall_accuracy': results['overall_accuracy'],
                'total_correct': results['total_correct'],
                'total_predictions': results['total_predictions'],
                'evaluation_time': results['evaluation_time'],
                'piece_accuracy': dict(results['piece_accuracy']),
                'piece_predictions': dict(results['piece_predictions']),
                'confusion_matrix': {k: dict(v) for k, v in results['confusion_matrix'].items()}
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")

def main():
    evaluator = PieceClassifierEvaluator()
    
    if not evaluator.test_images:
        logger.error("No test images found. Please check the dataset path.")
        return
    
    evaluator.load_models()
    
    if not evaluator.models:
        logger.error("No models loaded. Please check model paths.")
        return
    
    evaluator.evaluate_all_models()
    best_model, best_score = evaluator.find_best_model()
    evaluator.save_results()
    
    print(f"\n🎯 RECOMMENDATION: Use {best_model} as your piece classifier")
    print(f"   Expected real-world accuracy: {evaluator.results[best_model]['overall_accuracy']:.1%}")
    print(f"   Diversity score: {len(set(evaluator.results[best_model]['piece_predictions'].keys()))}/12 piece types")

if __name__ == "__main__":
    main()
