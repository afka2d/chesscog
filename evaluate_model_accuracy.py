#!/usr/bin/env python3
"""
Comprehensive model accuracy evaluation for real-world chess images.
Evaluates occupancy, color classification, piece classification, and combined accuracy.
"""

import os
import json
import logging
import time
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
import requests
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import chess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = {
            'occupancy': {'correct': 0, 'total': 0, 'details': []},
            'color': {'correct': 0, 'total': 0, 'details': []},
            'piece_type': {'correct': 0, 'total': 0, 'details': []},
            'combined': {'correct': 0, 'total': 0, 'details': []}
        }
        
        # Load models locally for detailed evaluation
        self.load_models()
        
    def load_models(self):
        """Load models for local evaluation"""
        logger.info("Loading models for evaluation...")
        
        # Load occupancy model
        occupancy_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        self.occupancy_model = torch.load(str(occupancy_model_path), map_location='cpu', weights_only=False)
        self.occupancy_model.eval()
        
        # Load color model
        from torchvision import models
        import torch.nn as nn
        
        def _get_color_model_architecture(num_classes):
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            return model
        
        def _get_piece_type_model_architecture(num_classes):
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            return model
        
        color_model_path = Path("models/color_classifier_simple.pt")
        self.color_model = _get_color_model_architecture(2)
        self.color_model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        self.color_model.eval()
        
        # Load piece type model
        piece_type_model_path = Path("models/piece_classifier_simple.pt")
        self.piece_type_model = _get_piece_type_model_architecture(6)
        self.piece_type_model.load_state_dict(torch.load(str(piece_type_model_path), map_location='cpu'))
        self.piece_type_model.eval()
        
        # Labels
        self.COLOR_LABELS = {0: "white", 1: "black"}
        self.PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}
        
        logger.info("✅ All models loaded successfully")
    
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
    
    def classify_square(self, square_img):
        """Classify a single square using all models"""
        # Transforms
        occupancy_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        color_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        piece_type_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        results = {}
        
        # Occupancy classification
        input_tensor = occupancy_transform(Image.fromarray(square_img)).unsqueeze(0)
        with torch.no_grad():
            occupancy_output = self.occupancy_model(input_tensor)
            probs = torch.softmax(occupancy_output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        results['occupancy'] = {
            'prediction': prediction,
            'confidence': confidence,
            'is_occupied': prediction == 1 and confidence > 0.5
        }
        
        if results['occupancy']['is_occupied']:
            # Color classification
            input_tensor_color = color_transform(Image.fromarray(square_img)).unsqueeze(0)
            with torch.no_grad():
                color_output = self.color_model(input_tensor_color)
                color_probs = torch.softmax(color_output, dim=1)
                color_confidence = torch.max(color_probs).item()
                color_prediction = torch.argmax(color_output, dim=1).item()
            
            results['color'] = {
                'prediction': color_prediction,
                'confidence': color_confidence,
                'label': self.COLOR_LABELS[color_prediction]
            }
            
            # Piece type classification
            input_tensor_piece = piece_type_transform(Image.fromarray(square_img)).unsqueeze(0)
            with torch.no_grad():
                piece_output = self.piece_type_model(input_tensor_piece)
                piece_probs = torch.softmax(piece_output, dim=1)
                piece_confidence = torch.max(piece_probs).item()
                piece_prediction = torch.argmax(piece_output, dim=1).item()
            
            results['piece_type'] = {
                'prediction': piece_prediction,
                'confidence': piece_confidence,
                'label': self.PIECE_TYPE_LABELS[piece_prediction]
            }
        else:
            results['color'] = None
            results['piece_type'] = None
        
        return results
    
    def load_ground_truth(self, image_path):
        """Load ground truth annotations for an image"""
        # Look for annotation file
        annotation_path = image_path.replace('.JPG', '.json').replace('.jpg', '.json')
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                return json.load(f)
        
        # If no annotation file, return None (will skip this image)
        return None
    
    def evaluate_image(self, image_path):
        """Evaluate a single image"""
        logger.info(f"Evaluating: {image_path}")
        
        # Load ground truth
        ground_truth = self.load_ground_truth(image_path)
        if ground_truth is None:
            logger.warning(f"No ground truth found for {image_path}, skipping...")
            return None
        
        # Load and process image
        img = cv2.imread(image_path)
        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect corners
        corners = self.detect_chessboard_corners(image_path)
        
        # Warp chessboard
        warped_board = self.warp_chessboard(img_array, corners)
        
        # Process each square
        image_results = {
            'occupancy': {'correct': 0, 'total': 0, 'details': []},
            'color': {'correct': 0, 'total': 0, 'details': []},
            'piece_type': {'correct': 0, 'total': 0, 'details': []},
            'combined': {'correct': 0, 'total': 0, 'details': []}
        }
        
        for rank in range(8):
            for file in range(8):
                square_img = self.extract_square(warped_board, rank, file)
                square_name = f"{chr(97+file)}{8-rank}"
                
                # Get ground truth for this square
                gt_square = ground_truth.get(square_name, {})
                gt_occupied = gt_square.get('occupied', False)
                gt_color = gt_square.get('color', None)
                gt_piece = gt_square.get('piece', None)
                
                # Classify square
                classification = self.classify_square(square_img)
                
                # Evaluate occupancy
                pred_occupied = classification['occupancy']['is_occupied']
                image_results['occupancy']['total'] += 1
                if pred_occupied == gt_occupied:
                    image_results['occupancy']['correct'] += 1
                
                image_results['occupancy']['details'].append({
                    'square': square_name,
                    'predicted': pred_occupied,
                    'ground_truth': gt_occupied,
                    'confidence': classification['occupancy']['confidence']
                })
                
                # Evaluate color and piece type if occupied
                if gt_occupied and pred_occupied:
                    # Color evaluation
                    if gt_color and classification['color']:
                        pred_color = classification['color']['label']
                        image_results['color']['total'] += 1
                        if pred_color == gt_color:
                            image_results['color']['correct'] += 1
                        
                        image_results['color']['details'].append({
                            'square': square_name,
                            'predicted': pred_color,
                            'ground_truth': gt_color,
                            'confidence': classification['color']['confidence']
                        })
                    
                    # Piece type evaluation
                    if gt_piece and classification['piece_type']:
                        pred_piece = classification['piece_type']['label']
                        image_results['piece_type']['total'] += 1
                        if pred_piece == gt_piece:
                            image_results['piece_type']['correct'] += 1
                        
                        image_results['piece_type']['details'].append({
                            'square': square_name,
                            'predicted': pred_piece,
                            'ground_truth': gt_piece,
                            'confidence': classification['piece_type']['confidence']
                        })
                    
                    # Combined evaluation (all three correct)
                    if (gt_color and gt_piece and 
                        classification['color'] and classification['piece_type'] and
                        pred_color == gt_color and pred_piece == gt_piece):
                        image_results['combined']['correct'] += 1
                    
                    image_results['combined']['total'] += 1
        
        return image_results
    
    def evaluate_dataset(self, dataset_path):
        """Evaluate entire dataset"""
        logger.info(f"Starting evaluation of dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(images)} images to evaluate")
        
        # Evaluate each image
        for i, image_path in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}: {image_path.name}")
            
            image_results = self.evaluate_image(str(image_path))
            if image_results is None:
                continue
            
            # Accumulate results
            for metric in ['occupancy', 'color', 'piece_type', 'combined']:
                self.results[metric]['correct'] += image_results[metric]['correct']
                self.results[metric]['total'] += image_results[metric]['total']
                self.results[metric]['details'].extend(image_results[metric]['details'])
        
        # Calculate final accuracies
        self.calculate_accuracies()
        self.generate_report()
    
    def calculate_accuracies(self):
        """Calculate final accuracies"""
        self.accuracies = {}
        for metric in ['occupancy', 'color', 'piece_type', 'combined']:
            if self.results[metric]['total'] > 0:
                accuracy = self.results[metric]['correct'] / self.results[metric]['total']
                self.accuracies[metric] = accuracy
            else:
                self.accuracies[metric] = 0.0
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("=" * 60)
        logger.info("MODEL ACCURACY EVALUATION REPORT")
        logger.info("=" * 60)
        
        for metric in ['occupancy', 'color', 'piece_type', 'combined']:
            accuracy = self.accuracies[metric]
            correct = self.results[metric]['correct']
            total = self.results[metric]['total']
            
            logger.info(f"{metric.upper()} ACCURACY: {accuracy:.3f} ({correct}/{total})")
        
        # Detailed analysis
        self.analyze_confidence_distributions()
        self.analyze_error_patterns()
        
        # Save detailed results
        self.save_results()
    
    def analyze_confidence_distributions(self):
        """Analyze confidence score distributions"""
        logger.info("\nCONFIDENCE ANALYSIS:")
        logger.info("-" * 30)
        
        for metric in ['occupancy', 'color', 'piece_type']:
            if metric in self.results and self.results[metric]['details']:
                confidences = [d['confidence'] for d in self.results[metric]['details']]
                logger.info(f"{metric.upper()}:")
                logger.info(f"  Mean confidence: {np.mean(confidences):.3f}")
                logger.info(f"  Std confidence: {np.std(confidences):.3f}")
                logger.info(f"  Min confidence: {np.min(confidences):.3f}")
                logger.info(f"  Max confidence: {np.max(confidences):.3f}")
    
    def analyze_error_patterns(self):
        """Analyze common error patterns"""
        logger.info("\nERROR PATTERN ANALYSIS:")
        logger.info("-" * 30)
        
        # Color confusion matrix
        if self.results['color']['details']:
            color_predictions = [d['predicted'] for d in self.results['color']['details']]
            color_ground_truth = [d['ground_truth'] for d in self.results['color']['details']]
            
            cm = confusion_matrix(color_ground_truth, color_predictions, labels=['white', 'black'])
            logger.info("Color Confusion Matrix:")
            logger.info("        Predicted")
            logger.info("        White  Black")
            logger.info(f"White   {cm[0,0]:4d}   {cm[0,1]:4d}")
            logger.info(f"Black   {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Piece type confusion matrix
        if self.results['piece_type']['details']:
            piece_predictions = [d['predicted'] for d in self.results['piece_type']['details']]
            piece_ground_truth = [d['ground_truth'] for d in self.results['piece_type']['details']]
            
            piece_labels = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
            cm = confusion_matrix(piece_ground_truth, piece_predictions, labels=piece_labels)
            
            logger.info("\nPiece Type Confusion Matrix:")
            logger.info("        " + " ".join([f"{label:>6}" for label in piece_labels]))
            for i, true_label in enumerate(piece_labels):
                row = f"{true_label:>6} " + " ".join([f"{cm[i,j]:6d}" for j in range(len(piece_labels))])
                logger.info(row)
    
    def save_results(self):
        """Save detailed results to file"""
        results_file = "model_evaluation_results.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'accuracies': self.accuracies,
            'results': {
                metric: {
                    'correct': self.results[metric]['correct'],
                    'total': self.results[metric]['total'],
                    'accuracy': self.accuracies[metric]
                }
                for metric in ['occupancy', 'color', 'piece_type', 'combined']
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {results_file}")

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    # Evaluate on your training images
    dataset_path = "my_chess_images/train/images"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        logger.info("Please ensure you have chess images with ground truth annotations")
        return
    
    evaluator.evaluate_dataset(dataset_path)

if __name__ == "__main__":
    main()
