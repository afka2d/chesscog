#!/usr/bin/env python3
"""
Validate Marshall-trained models against real-world examples
Test for overfitting by evaluating on unseen data and comparing with original models
"""

import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import time
from sklearn.model_selection import train_test_split
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_square_for_occupancy(square):
    """Preprocess square for occupancy detection"""
    square = cv2.resize(square, (100, 100))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square = square.astype(np.float32) / 255.0
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def preprocess_square_for_color(square):
    """Preprocess square for color classification"""
    square = cv2.resize(square, (224, 224))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square = square.astype(np.float32) / 255.0
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def preprocess_square_for_piece(square):
    """Preprocess square for piece classification"""
    square = cv2.resize(square, (224, 224))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square = square.astype(np.float32) / 255.0
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def warp_board(image, corners):
    """Warp image to get a square chessboard"""
    try:
        src_points = np.array(corners, dtype=np.float32)
        size = 400
        dst_points = np.array([
            [0, 0], [size, 0], [size, size], [0, size]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (size, size))
        return warped
    except Exception as e:
        logger.warning(f"Error warping board: {e}")
        return None

def fen_to_board(fen):
    """Convert FEN string to 8x8 board representation"""
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    parts = fen.split()
    if not parts:
        return board
    
    ranks = parts[0].split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                if file_idx < 8:
                    board[rank_idx][file_idx] = char
                    file_idx += 1
    
    return board

def extract_squares_with_labels(warped_board, fen, task_type):
    """Extract squares and create labels based on task type"""
    squares = []
    labels = []
    
    board = fen_to_board(fen)
    square_size = warped_board.shape[0] // 8
    
    for rank in range(8):
        for file in range(8):
            y1 = rank * square_size
            y2 = (rank + 1) * square_size
            x1 = file * square_size
            x2 = (file + 1) * square_size
            
            square = warped_board[y1:y2, x1:x2]
            squares.append(square)
            
            piece = board[rank][file]
            
            if task_type == "occupancy":
                # 1 if occupied, 0 if empty
                label = 1 if piece != '.' else 0
                labels.append(label)
            elif task_type == "color":
                if piece == '.':
                    labels.append(None)  # Skip empty squares
                else:
                    # 0 for white (uppercase), 1 for black (lowercase)
                    label = 0 if piece.isupper() else 1
                    labels.append(label)
            elif task_type == "piece":
                if piece == '.':
                    labels.append(None)  # Skip empty squares
                else:
                    piece_map = {
                        'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
                        'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11
                    }
                    label = piece_map.get(piece, 12)
                    labels.append(label)
    
    return squares, labels

def load_model(model_path, model_type):
    """Load a trained model"""
    try:
        if model_type == "occupancy":
            # For Marshall occupancy model, it's saved as state_dict
            # We need to load the original model architecture first
            original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
            if original_model_path.exists():
                # Load the original model architecture
                model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
                # Load the Marshall weights
                marshall_weights = torch.load(str(model_path), map_location='cpu', weights_only=True)
                model.load_state_dict(marshall_weights)
            else:
                logger.error(f"Original occupancy model not found: {original_model_path}")
                return None
        else:
            # For color and piece, we need to create the architecture first
            if model_type == "color":
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            elif model_type == "piece":
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 13)
            
            model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
        
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

def create_test_dataset(task_type, test_split=0.3):
    """Create a test dataset from Marshall annotations"""
    logger.info(f"Creating {task_type} test dataset...")
    
    # Load Marshall annotations
    with open("marshall_chess_annotations/annotations.json", 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    
    # Filter out excluded images
    valid_annotations = {
        k: v for k, v in annotations.items() 
        if k not in excluded_images
    }
    
    # Split into train/test (use same random state as training)
    image_names = list(valid_annotations.keys())
    train_names, test_names = train_test_split(
        image_names, test_size=test_split, random_state=42
    )
    
    logger.info(f"Using {len(test_names)} images for testing")
    
    dataset = []
    marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    
    processed = 0
    errors = 0
    
    for image_name in test_names:
        image_path = marshall_photos_dir / image_name
        if not image_path.exists():
            continue
            
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
                continue
                
            annotation = valid_annotations[image_name]
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                continue
            
            # Warp board
            warped_board = warp_board(image, corners)
            if warped_board is None:
                continue
            
            # Extract squares and labels
            squares, labels = extract_squares_with_labels(warped_board, fen, task_type)
            
            for square, label in zip(squares, labels):
                if square is not None and label is not None:
                    if task_type == "occupancy":
                        square_tensor = preprocess_square_for_occupancy(square)
                    elif task_type == "color":
                        square_tensor = preprocess_square_for_color(square)
                    elif task_type == "piece":
                        square_tensor = preprocess_square_for_piece(square)
                    
                    dataset.append({
                        'square': square_tensor,
                        'label': torch.tensor(label, dtype=torch.long),
                        'image_name': image_name
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} test images...")
    
    logger.info(f"Created {task_type} test dataset with {len(dataset)} samples")
    logger.info(f"Processed: {processed}, Errors: {errors}")
    return dataset

def evaluate_model(model, test_data, task_type, model_name):
    """Evaluate a model on test data"""
    logger.info(f"Evaluating {model_name} on {task_type} task...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sample in test_data:
            square = sample['square'].unsqueeze(0).to(device)
            label = sample['label'].to(device)
            
            output = model(square)
            _, predicted = torch.max(output.data, 1)
            
            total += 1
            correct += (predicted == label).sum().item()
            
            all_predictions.append(predicted.cpu().item())
            all_labels.append(label.cpu().item())
    
    accuracy = 100 * correct / total
    logger.info(f"{model_name} - {task_type}: {accuracy:.2f}% accuracy ({correct}/{total})")
    
    return accuracy, all_predictions, all_labels

def compare_models(task_type):
    """Compare original and Marshall-trained models"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing {task_type} models")
    logger.info(f"{'='*60}")
    
    # Create test dataset
    test_data = create_test_dataset(task_type)
    if len(test_data) < 100:
        logger.error(f"Not enough test data for {task_type}")
        return
    
    # Load models
    if task_type == "occupancy":
        original_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    elif task_type == "color":
        original_path = Path("models/color_classifier_simple.pt")
        marshall_path = Path("models_marshall_improved/color_classification_marshall.pt")
    elif task_type == "piece":
        original_path = Path("models/piece_classifier_simple.pt")
        marshall_path = Path("models_marshall_improved/piece_classification_marshall.pt")
    
    # Check if models exist
    if not original_path.exists():
        logger.error(f"Original {task_type} model not found: {original_path}")
        return
    
    if not marshall_path.exists():
        logger.error(f"Marshall {task_type} model not found: {marshall_path}")
        return
    
    # Load models
    original_model = load_model(original_path, task_type)
    marshall_model = load_model(marshall_path, task_type)
    
    if original_model is None or marshall_model is None:
        logger.error("Failed to load models")
        return
    
    # Evaluate both models
    orig_acc, orig_pred, orig_labels = evaluate_model(
        original_model, test_data, task_type, "Original"
    )
    
    marshall_acc, marshall_pred, marshall_labels = evaluate_model(
        marshall_model, test_data, task_type, "Marshall"
    )
    
    # Calculate improvement
    improvement = marshall_acc - orig_acc
    logger.info(f"\n📊 Results Summary:")
    logger.info(f"Original model:  {orig_acc:.2f}%")
    logger.info(f"Marshall model:  {marshall_acc:.2f}%")
    logger.info(f"Improvement:     {improvement:+.2f}%")
    
    if improvement > 0:
        logger.info(f"✅ Marshall model is {improvement:.2f}% better!")
    elif improvement < -1:
        logger.warning(f"⚠️  Marshall model is {abs(improvement):.2f}% worse - possible overfitting")
    else:
        logger.info(f"ℹ️  Models perform similarly")
    
    return {
        'task': task_type,
        'original_accuracy': orig_acc,
        'marshall_accuracy': marshall_acc,
        'improvement': improvement,
        'test_samples': len(test_data)
    }

def main():
    """Run comprehensive validation of Marshall models"""
    logger.info("🔍 Starting Marshall Model Validation")
    logger.info("Testing for overfitting and real-world performance")
    logger.info("=" * 60)
    
    # Check if Marshall models exist
    marshall_dir = Path("models_marshall_improved")
    if not marshall_dir.exists():
        logger.error("Marshall models directory not found!")
        return
    
    # List available models
    available_models = list(marshall_dir.glob("*.pt"))
    logger.info(f"Found Marshall models: {[m.name for m in available_models]}")
    
    results = []
    
    # Test each model type (only test available models)
    available_tasks = []
    if Path("models_marshall_improved/occupancy_marshall.pt").exists():
        available_tasks.append("occupancy")
    if Path("models_marshall_improved/color_classification_marshall.pt").exists():
        available_tasks.append("color")
    if Path("models_marshall_improved/piece_classification_marshall.pt").exists():
        available_tasks.append("piece")
    
    logger.info(f"Testing available models: {available_tasks}")
    
    for task_type in available_tasks:
        try:
            result = compare_models(task_type)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error testing {task_type}: {e}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("🏁 VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    if results:
        for result in results:
            task = result['task']
            orig = result['original_accuracy']
            marshall = result['marshall_accuracy']
            improvement = result['improvement']
            samples = result['test_samples']
            
            status = "✅ BETTER" if improvement > 0 else "⚠️  WORSE" if improvement < -1 else "ℹ️  SIMILAR"
            
            logger.info(f"{task.upper():>12}: {orig:6.2f}% → {marshall:6.2f}% ({improvement:+6.2f}%) {status} ({samples} samples)")
        
        # Overall assessment
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        if avg_improvement > 1:
            logger.info(f"\n🎉 Overall: Marshall models show {avg_improvement:.2f}% average improvement!")
        elif avg_improvement < -1:
            logger.warning(f"\n⚠️  Overall: Marshall models show {abs(avg_improvement):.2f}% average degradation - possible overfitting")
        else:
            logger.info(f"\nℹ️  Overall: Marshall models perform similarly to originals")
    else:
        logger.error("No validation results available")

if __name__ == "__main__":
    main()
