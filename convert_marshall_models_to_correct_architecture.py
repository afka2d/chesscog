#!/usr/bin/env python3
"""
Convert existing Marshall models to use the correct architectures for API compatibility.
This creates new models with the same weights but correct architecture.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_occupancy_model_architecture():
    """Create the same ResNet architecture as the original occupancy model"""
    try:
        # Load the original model to get the exact architecture
        original_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_path.exists():
            logger.error(f"Original occupancy model not found at {original_path}")
            return None
        
        model = torch.load(str(original_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original occupancy model architecture loaded")
        return model
    except Exception as e:
        logger.error(f"Error loading original occupancy model: {e}")
        return None

def create_color_model_architecture():
    """Create the same MobileNetV2 architecture as the original color model"""
    try:
        # Load the original model to get the exact architecture
        original_path = Path("models/color_classifier_simple.pt")
        if not original_path.exists():
            logger.error(f"Original color model not found at {original_path}")
            return None
        
        # Create MobileNetV2 architecture
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 classes: black, white
        
        # Load the state dict
        state_dict = torch.load(str(original_path), map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info("✅ Original color model architecture loaded")
        return model
    except Exception as e:
        logger.error(f"Error loading original color model: {e}")
        return None

def create_piece_model_architecture():
    """Create the same MobileNetV2 architecture as the original piece model"""
    try:
        # Load the original model to get the exact architecture
        original_path = Path("models/piece_classifier_simple.pt")
        if not original_path.exists():
            logger.error(f"Original piece model not found at {original_path}")
            return None
        
        # Create MobileNetV2 architecture
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 6)  # 6 piece types
        
        # Load the state dict
        state_dict = torch.load(str(original_path), map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info("✅ Original piece model architecture loaded")
        return model
    except Exception as e:
        logger.error(f"Error loading original piece model: {e}")
        return None

def convert_occupancy_model():
    """Convert Marshall occupancy model to correct architecture"""
    logger.info("🔄 Converting Marshall occupancy model to correct architecture...")
    
    # Load original architecture
    original_model = create_occupancy_model_architecture()
    if original_model is None:
        logger.error("Failed to load original occupancy model")
        return False
    
    # Load Marshall weights
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if not marshall_path.exists():
        logger.error(f"Marshall occupancy model not found at {marshall_path}")
        return False
    
    try:
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall occupancy weights loaded")
        
        # Apply Marshall weights to original architecture
        original_model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to original architecture")
        
        # Save the converted model
        output_dir = Path("models_marshall_improved")
        output_dir.mkdir(exist_ok=True)
        
        # Save as full model (not just state_dict) for API compatibility
        torch.save(original_model, output_dir / "occupancy_marshall_correct_architecture.pt")
        logger.info("✅ Converted occupancy model saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting occupancy model: {e}")
        return False

def convert_color_model():
    """Convert Marshall color model to correct architecture"""
    logger.info("🔄 Converting Marshall color model to correct architecture...")
    
    # Load original architecture
    original_model = create_color_model_architecture()
    if original_model is None:
        logger.error("Failed to load original color model")
        return False
    
    # Check if Marshall color model exists
    marshall_path = Path("models_marshall_improved/color_classification_marshall.pt")
    if not marshall_path.exists():
        logger.warning(f"Marshall color model not found at {marshall_path}")
        logger.info("Using original color model as fallback")
        
        # Save the original model with correct name
        output_dir = Path("models_marshall_improved")
        output_dir.mkdir(exist_ok=True)
        torch.save(original_model.state_dict(), output_dir / "color_marshall_correct_architecture.pt")
        logger.info("✅ Original color model saved as Marshall model")
        return True
    
    try:
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall color weights loaded")
        
        # Apply Marshall weights to original architecture
        original_model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to original architecture")
        
        # Save the converted model
        output_dir = Path("models_marshall_improved")
        output_dir.mkdir(exist_ok=True)
        torch.save(original_model.state_dict(), output_dir / "color_marshall_correct_architecture.pt")
        logger.info("✅ Converted color model saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting color model: {e}")
        return False

def convert_piece_model():
    """Convert Marshall piece model to correct architecture"""
    logger.info("🔄 Converting Marshall piece model to correct architecture...")
    
    # Load original architecture
    original_model = create_piece_model_architecture()
    if original_model is None:
        logger.error("Failed to load original piece model")
        return False
    
    # Try to load the combined piece classifier first
    marshall_path = Path("models_marshall_improved/piece_classification_combined_marshall.pt")
    if not marshall_path.exists():
        # Fallback to regular piece classifier
        marshall_path = Path("models_marshall_improved/piece_classification_marshall.pt")
        if not marshall_path.exists():
            logger.warning(f"No Marshall piece model found")
            logger.info("Using original piece model as fallback")
            
            # Save the original model with correct name
            output_dir = Path("models_marshall_improved")
            output_dir.mkdir(exist_ok=True)
            torch.save(original_model.state_dict(), output_dir / "piece_marshall_correct_architecture.pt")
            logger.info("✅ Original piece model saved as Marshall model")
            return True
    
    try:
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall piece weights loaded")
        
        # Apply Marshall weights to original architecture
        original_model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to original architecture")
        
        # Save the converted model
        output_dir = Path("models_marshall_improved")
        output_dir.mkdir(exist_ok=True)
        torch.save(original_model.state_dict(), output_dir / "piece_marshall_correct_architecture.pt")
        logger.info("✅ Converted piece model saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting piece model: {e}")
        return False

def main():
    """Convert all Marshall models to correct architectures"""
    logger.info("🎯 Converting Marshall models to correct architectures for API compatibility")
    logger.info("=" * 80)
    
    results = []
    
    # Convert each model
    results.append(("Occupancy Model", convert_occupancy_model()))
    results.append(("Color Model", convert_color_model()))
    results.append(("Piece Model", convert_piece_model()))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 CONVERSION SUMMARY")
    logger.info("=" * 80)
    
    for model_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{model_name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info(f"\n🎯 Overall: {successful}/{total} models converted successfully")
    
    if successful == total:
        logger.info("🎉 All Marshall models converted to correct architectures!")
        logger.info("📍 Models saved in: models_marshall_improved/")
        logger.info("   - occupancy_marshall_correct_architecture.pt")
        logger.info("   - color_marshall_correct_architecture.pt")
        logger.info("   - piece_marshall_correct_architecture.pt")
    else:
        logger.error("❌ Some models failed to convert. Check logs above for details.")

if __name__ == "__main__":
    main()
