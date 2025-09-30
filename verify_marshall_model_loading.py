#!/usr/bin/env python3
"""
Verify which occupancy model the Marshall API is actually loading
and test it on sample images with visual output.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import logging
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_marshall_occupancy_model():
    """Load the Marshall occupancy model (same method as API)."""
    try:
        # Load the original model architecture first
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            logger.error(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original occupancy model architecture loaded")
        
        # Load the Marshall weights (state_dict)
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            logger.error(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall occupancy weights loaded")
        
        # Apply the Marshall weights to the original model architecture
        model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to model")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading Marshall occupancy model: {e}")
        return None

def preprocess_square_for_occupancy(square_img):
    """Preprocess square for occupancy detection (same as API)."""
    square = cv2.resize(square_img, (100, 100))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square = square.astype(np.float32) / 255.0
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def load_original_occupancy_model():
    """Load the original occupancy model for comparison."""
    try:
        model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not model_path.exists():
            logger.error(f"❌ Original occupancy model not found at {model_path}")
            return None
        
        model = torch.load(str(model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original occupancy model loaded")
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading original occupancy model: {e}")
        return None

def test_model_on_image(model, image_path, corners, model_name):
    """Test a model on a single image and return predictions."""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"❌ Could not load image: {image_path}")
            return None
        
        # Warp board
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
            np.array(corners, dtype=np.float32),
            np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
        ), (800, 800))
        
        # Extract squares and make predictions
        predictions = []
        square_size = 100
        
        for rank in range(8):
            for file in range(8):
                # Extract square
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped[y1:y2, x1:x2]
                
                # Preprocess
                input_tensor = preprocess_square_for_occupancy(square).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.softmax(output, dim=1)[0][prediction].item()
                
                predictions.append({
                    'rank': rank,
                    'file': file,
                    'square': square,
                    'prediction': prediction,
                    'confidence': confidence,
                    'occupied': prediction == 1
                })
        
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error testing model on image: {e}")
        return None

def visualize_predictions(image_path, corners, marshall_predictions, original_predictions, output_path):
    """Create a visualization comparing both models' predictions."""
    try:
        # Load original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Comparison: {Path(image_path).name}', fontsize=16)
        
        # Original image with corners
        axes[0, 0].imshow(img_rgb)
        corners_array = np.array(corners)
        axes[0, 0].plot(corners_array[:, 0], corners_array[:, 1], 'ro-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Original Image with Corners')
        axes[0, 0].axis('off')
        
        # Warped board
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
            np.array(corners, dtype=np.float32),
            np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
        ), (800, 800))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(warped_rgb)
        axes[0, 1].set_title('Warped Board')
        axes[0, 1].axis('off')
        
        # Marshall model predictions
        marshall_board = np.zeros((8, 8), dtype=int)
        for pred in marshall_predictions:
            marshall_board[pred['rank'], pred['file']] = 1 if pred['occupied'] else 0
        
        im1 = axes[0, 2].imshow(marshall_board, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 2].set_title('Marshall Model Predictions\n(Red=Empty, Green=Occupied)')
        axes[0, 2].set_xlabel('File (a-h)')
        axes[0, 2].set_ylabel('Rank (1-8)')
        axes[0, 2].set_xticks(range(8))
        axes[0, 2].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[0, 2].set_yticks(range(8))
        axes[0, 2].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Original model predictions
        original_board = np.zeros((8, 8), dtype=int)
        for pred in original_predictions:
            original_board[pred['rank'], pred['file']] = 1 if pred['occupied'] else 0
        
        im2 = axes[1, 0].imshow(original_board, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 0].set_title('Original Model Predictions\n(Red=Empty, Green=Occupied)')
        axes[1, 0].set_xlabel('File (a-h)')
        axes[1, 0].set_ylabel('Rank (1-8)')
        axes[1, 0].set_xticks(range(8))
        axes[1, 0].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1, 0].set_yticks(range(8))
        axes[1, 0].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Difference
        diff_board = marshall_board - original_board
        im3 = axes[1, 1].imshow(diff_board, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title('Difference\n(Blue=Marshall Empty, Red=Original Empty)')
        axes[1, 1].set_xlabel('File (a-h)')
        axes[1, 1].set_ylabel('Rank (1-8)')
        axes[1, 1].set_xticks(range(8))
        axes[1, 1].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1, 1].set_yticks(range(8))
        axes[1, 1].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        
        # Statistics
        marshall_occupied = np.sum(marshall_board)
        original_occupied = np.sum(original_board)
        differences = np.sum(np.abs(diff_board))
        
        stats_text = f"""Statistics:
Marshall Model: {marshall_occupied} occupied squares
Original Model: {original_occupied} occupied squares
Differences: {differences} squares
Agreement: {100 * (64 - differences) / 64:.1f}%"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Comparison Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating visualization: {e}")

def main():
    logger.info("🔍 Verifying Marshall API Model Loading and Testing")
    logger.info("=" * 60)
    
    # Load models
    logger.info("Loading Marshall occupancy model...")
    marshall_model = load_marshall_occupancy_model()
    if marshall_model is None:
        logger.error("❌ Failed to load Marshall model")
        return
    
    logger.info("Loading original occupancy model...")
    original_model = load_original_occupancy_model()
    if original_model is None:
        logger.error("❌ Failed to load original model")
        return
    
    # Test on some sample images
    test_images = [
        {
            'path': 'yolo_detection_IMG_4763.jpg',
            'corners': [[578.0, 1939.0], [2628.0, 1889.0], [2791.0, 4042.0], [397.0, 4025.0]]
        },
        {
            'path': 'yolo_detection_IMG_4764.jpg', 
            'corners': [[662.0, 1972.0], [2685.0, 1850.0], [2808.0, 4046.0], [439.0, 4010.0]]
        }
    ]
    
    for i, test_case in enumerate(test_images):
        image_path = Path(test_case['path'])
        if not image_path.exists():
            logger.warning(f"⚠️ Test image not found: {image_path}")
            continue
        
        logger.info(f"\n📸 Testing image {i+1}: {image_path.name}")
        
        # Test Marshall model
        marshall_predictions = test_model_on_image(
            marshall_model, image_path, test_case['corners'], "Marshall"
        )
        
        # Test original model
        original_predictions = test_model_on_image(
            original_model, image_path, test_case['corners'], "Original"
        )
        
        if marshall_predictions and original_predictions:
            # Count occupied squares
            marshall_occupied = sum(1 for p in marshall_predictions if p['occupied'])
            original_occupied = sum(1 for p in original_predictions if p['occupied'])
            
            logger.info(f"   Marshall model: {marshall_occupied} occupied squares")
            logger.info(f"   Original model: {original_occupied} occupied squares")
            
            # Create visualization
            output_path = f"model_comparison_{i+1}_{image_path.stem}.png"
            visualize_predictions(
                image_path, test_case['corners'], 
                marshall_predictions, original_predictions, output_path
            )
    
    logger.info("\n✅ Model verification complete!")
    logger.info("Check the generated PNG files for visual comparisons.")

if __name__ == "__main__":
    main()
