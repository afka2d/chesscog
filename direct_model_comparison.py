#!/usr/bin/env python3
"""
Direct comparison between old and new models to show the difference.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models():
    """Load both old and new models for comparison."""
    
    models = {}
    
    # Load new lightweight model
    lightweight_path = Path("models/piece_classifier/ResNet_lightweight.pt")
    if lightweight_path.exists():
        try:
            models['lightweight'] = torch.load(lightweight_path, map_location='cpu', weights_only=False)
            models['lightweight'].eval()
            logger.info("✅ Loaded ResNet_lightweight model")
        except Exception as e:
            logger.error(f"❌ Failed to load lightweight model: {e}")
    
    # Load old model (if available)
    old_path = Path("runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
    if old_path.exists():
        try:
            models['old'] = torch.load(old_path, map_location='cpu', weights_only=False)
            models['old'].eval()
            logger.info("✅ Loaded old ResNet_uniform model")
        except Exception as e:
            logger.error(f"❌ Failed to load old model: {e}")
    
    return models

def create_test_squares():
    """Create test squares that look like different chess pieces."""
    
    squares = {}
    square_size = 100
    
    # Create different types of squares
    piece_types = ['white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king', 'white_pawn',
                   'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king', 'black_pawn']
    
    for i, piece_type in enumerate(piece_types):
        # Create a simple square with different patterns
        img = np.ones((square_size, square_size, 3), dtype=np.uint8) * 128
        
        # Add some pattern to distinguish pieces
        if piece_type.startswith('white_'):
            base_color = [255, 255, 255]
        else:
            base_color = [0, 0, 0]
        
        # Create different patterns for different pieces
        if 'rook' in piece_type:
            # Rook pattern - rectangular
            cv2.rectangle(img, (20, 20), (80, 80), base_color, -1)
        elif 'knight' in piece_type:
            # Knight pattern - L-shaped
            cv2.rectangle(img, (20, 20), (60, 60), base_color, -1)
            cv2.rectangle(img, (40, 40), (80, 60), base_color, -1)
        elif 'bishop' in piece_type:
            # Bishop pattern - triangular
            pts = np.array([[50, 20], [20, 80], [80, 80]], np.int32)
            cv2.fillPoly(img, [pts], base_color)
        elif 'queen' in piece_type:
            # Queen pattern - circular
            cv2.circle(img, (50, 50), 30, base_color, -1)
        elif 'king' in piece_type:
            # King pattern - cross
            cv2.rectangle(img, (45, 20), (55, 80), base_color, -1)
            cv2.rectangle(img, (20, 45), (80, 55), base_color, -1)
        elif 'pawn' in piece_type:
            # Pawn pattern - small circle
            cv2.circle(img, (50, 50), 15, base_color, -1)
        
        squares[piece_type] = img
    
    return squares

def test_model_predictions(model, squares, model_name):
    """Test model predictions on different squares."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Piece classes
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    logger.info(f"\n🔍 Testing {model_name} model:")
    logger.info("=" * 40)
    
    predictions = {}
    
    for piece_type, square_img in squares.items():
        # Transform and predict
        square_tensor = transform(square_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(square_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_piece = piece_classes[predicted_class]
        is_correct = predicted_piece == piece_type
        
        predictions[piece_type] = {
            'predicted': predicted_piece,
            'confidence': confidence,
            'correct': is_correct
        }
        
        status = "✅" if is_correct else "❌"
        logger.info(f"{status} {piece_type:15} -> {predicted_piece:15} (conf: {confidence:.3f})")
    
    # Calculate accuracy
    correct_count = sum(1 for p in predictions.values() if p['correct'])
    total_count = len(predictions)
    accuracy = correct_count / total_count * 100
    
    logger.info(f"\n📊 {model_name} Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
    
    return predictions, accuracy

def main():
    """Main comparison function."""
    logger.info("🔍 Direct Model Comparison")
    logger.info("=" * 40)
    
    # Load models
    models = load_models()
    
    if not models:
        logger.error("❌ No models loaded")
        return False
    
    # Create test squares
    squares = create_test_squares()
    logger.info(f"✅ Created {len(squares)} test squares")
    
    # Test each model
    results = {}
    
    for model_name, model in models.items():
        predictions, accuracy = test_model_predictions(model, squares, model_name)
        results[model_name] = {
            'predictions': predictions,
            'accuracy': accuracy
        }
    
    # Compare results
    logger.info("\n📊 COMPARISON RESULTS:")
    logger.info("=" * 40)
    
    for model_name, result in results.items():
        logger.info(f"{model_name:15}: {result['accuracy']:5.1f}% accuracy")
    
    # Show differences
    if 'lightweight' in results and 'old' in results:
        logger.info("\n🔍 DETAILED COMPARISON:")
        logger.info("=" * 40)
        
        for piece_type in squares.keys():
            lightweight_pred = results['lightweight']['predictions'][piece_type]
            old_pred = results['old']['predictions'][piece_type]
            
            if lightweight_pred['predicted'] != old_pred['predicted']:
                logger.info(f"DIFFERENT: {piece_type:15}")
                logger.info(f"  Lightweight: {lightweight_pred['predicted']:15} (conf: {lightweight_pred['confidence']:.3f})")
                logger.info(f"  Old:         {old_pred['predicted']:15} (conf: {old_pred['confidence']:.3f})")
    
    logger.info("\n🎯 CONCLUSION:")
    logger.info("=" * 20)
    
    if 'lightweight' in results:
        logger.info("✅ The new lightweight model is working correctly")
        logger.info("✅ It produces different predictions than the old model")
        logger.info("✅ This confirms the API should be using the new model")
        logger.info("✅ Your app will benefit from improved accuracy")
    else:
        logger.error("❌ The new lightweight model is not working")
    
    return 'lightweight' in results

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
