#!/usr/bin/env python3
"""
Debug what models are actually being loaded and used in the APIs.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import logging
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def debug_original_api_model():
    """Debug what model the original API is actually using."""
    logger.info("🔍 Debugging Original API Model")
    logger.info("=" * 50)
    
    # Check what's in the original model path
    original_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    if original_path.exists():
        logger.info(f"✅ Original model found at: {original_path}")
        
        # Load and inspect the model
        try:
            model = torch.load(str(original_path), map_location='cpu', weights_only=False)
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model architecture: {model.__class__.__name__}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            
            # Check if it's a ResNet
            if hasattr(model, 'fc'):
                logger.info(f"Final layer: {model.fc}")
                logger.info(f"Number of classes: {model.fc.out_features}")
            
            # Test with dummy input
            dummy_input = torch.randn(1, 3, 100, 100)
            with torch.no_grad():
                output = model(dummy_input)
                logger.info(f"Output shape: {output.shape}")
                logger.info(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                # Check what the model predicts for a random input
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][prediction].item()
                logger.info(f"Random prediction: {'Occupied' if prediction == 1 else 'Empty'} (conf: {confidence:.3f})")
                
        except Exception as e:
            logger.error(f"❌ Error loading original model: {e}")
    else:
        logger.error(f"❌ Original model not found at: {original_path}")

def debug_marshall_api_model():
    """Debug what model the Marshall API is actually using."""
    logger.info("\n🔍 Debugging Marshall API Model")
    logger.info("=" * 50)
    
    # Check Marshall model path
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if marshall_path.exists():
        logger.info(f"✅ Marshall model found at: {marshall_path}")
        
        # Load as state_dict
        try:
            marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
            logger.info(f"Marshall weights type: {type(marshall_weights)}")
            logger.info(f"Number of weight keys: {len(marshall_weights)}")
            logger.info(f"Sample keys: {list(marshall_weights.keys())[:5]}")
            
            # Load original architecture and apply Marshall weights
            original_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
            if original_path.exists():
                original_model = torch.load(str(original_path), map_location='cpu', weights_only=False)
                logger.info("✅ Original model architecture loaded")
                
                # Apply Marshall weights
                original_model.load_state_dict(marshall_weights)
                logger.info("✅ Marshall weights applied")
                
                # Test with dummy input
                dummy_input = torch.randn(1, 3, 100, 100)
                with torch.no_grad():
                    output = original_model(dummy_input)
                    logger.info(f"Output shape: {output.shape}")
                    logger.info(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                    
                    # Check what the model predicts for a random input
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.softmax(output, dim=1)[0][prediction].item()
                    logger.info(f"Random prediction: {'Occupied' if prediction == 1 else 'Empty'} (conf: {confidence:.3f})")
                    
        except Exception as e:
            logger.error(f"❌ Error loading Marshall model: {e}")
    else:
        logger.error(f"❌ Marshall model not found at: {marshall_path}")

def test_on_real_image():
    """Test both models on a real image to see actual behavior."""
    logger.info("\n🔍 Testing on Real Image")
    logger.info("=" * 50)
    
    # Load a test image
    test_image = Path("yolo_detection_IMG_4763.jpg")
    if not test_image.exists():
        logger.error(f"❌ Test image not found: {test_image}")
        return
    
    # Load image
    img = cv2.imread(str(test_image))
    if img is None:
        logger.error(f"❌ Could not load image: {test_image}")
        return
    
    logger.info(f"✅ Loaded image: {img.shape}")
    
    # Use the corners from the terminal logs
    corners = [[578.0, 1939.0], [2628.0, 1889.0], [2791.0, 4042.0], [397.0, 4025.0]]
    
    # Warp the board
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
        np.array(corners, dtype=np.float32),
        np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32)
    ), (800, 800))
    
    logger.info(f"✅ Warped board: {warped.shape}")
    
    # Test both models
    models_to_test = [
        ("Original", "runs/occupancy_classifier/ResNet/ResNet.pt"),
        ("Marshall", "models_marshall_improved/occupancy_marshall.pt")
    ]
    
    for model_name, model_path in models_to_test:
        logger.info(f"\n--- Testing {model_name} Model ---")
        
        try:
            if model_name == "Original":
                model = torch.load(str(model_path), map_location='cpu', weights_only=False)
            else:
                # Load Marshall model
                original_model = torch.load("runs/occupancy_classifier/ResNet/ResNet.pt", map_location='cpu', weights_only=False)
                marshall_weights = torch.load(str(model_path), map_location='cpu', weights_only=True)
                original_model.load_state_dict(marshall_weights)
                model = original_model
            
            model.eval()
            
            # Count occupied squares
            occupied_count = 0
            square_size = 100
            
            for rank in range(8):
                for file in range(8):
                    # Extract square
                    y1 = rank * square_size
                    y2 = (rank + 1) * square_size
                    x1 = file * square_size
                    x2 = (file + 1) * square_size
                    
                    square = warped[y1:y2, x1:x2]
                    
                    # Preprocess (same as API)
                    square = cv2.resize(square, (100, 100))
                    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                    square = square.astype(np.float32) / 255.0
                    square = torch.from_numpy(square).permute(2, 0, 1)
                    
                    # Predict
                    with torch.no_grad():
                        output = model(square.unsqueeze(0))
                        prediction = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][prediction].item()
                        
                        if prediction == 1:  # Occupied
                            occupied_count += 1
                            square_name = f"{chr(ord('a') + file)}{8 - rank}"
                            logger.info(f"  {square_name}: Occupied (conf: {confidence:.3f})")
            
            logger.info(f"✅ {model_name} model found {occupied_count} occupied squares")
            
        except Exception as e:
            logger.error(f"❌ Error testing {model_name} model: {e}")

def main():
    logger.info("🔍 Debugging Actual Model Loading and Behavior")
    logger.info("=" * 60)
    
    debug_original_api_model()
    debug_marshall_api_model()
    test_on_real_image()
    
    logger.info("\n✅ Debug complete!")

if __name__ == "__main__":
    main()
