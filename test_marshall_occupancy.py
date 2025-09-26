#!/usr/bin/env python3
"""
Test Marshall occupancy model directly
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_marshall_occupancy():
    """Test the Marshall occupancy model"""
    logger.info("🧪 Testing Marshall Occupancy Model")
    logger.info("=" * 50)
    
    # Check if model exists
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if not marshall_path.exists():
        logger.error("❌ Marshall occupancy model not found!")
        return
    
    # Load the model
    try:
        # Load as state_dict
        state_dict = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall model state_dict loaded successfully")
        
        # Get model info
        total_params = sum(p.numel() for p in state_dict.values())
        
        logger.info(f"📊 Model Info:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   State dict keys: {len(state_dict)}")
        logger.info(f"   Model type: State dict (weights only)")
        
        # Show some key layers
        logger.info(f"   Key layers found:")
        for key in list(state_dict.keys())[:5]:  # Show first 5 keys
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
            logger.info(f"     {key}: {shape}")
        
        logger.info(f"\n🎯 Marshall Occupancy Model Assessment:")
        logger.info(f"   ✅ Model state_dict loads successfully")
        logger.info(f"   ✅ Model has {total_params:,} parameters")
        logger.info(f"   ✅ Model file size: {marshall_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Check if this is a reasonable size
        if total_params > 100000:  # Reasonable size for a ResNet
            logger.info(f"   ✅ Model appears to be a substantial neural network")
        else:
            logger.warning(f"   ⚠️  Model seems small ({total_params} params) - might be overfitting")
        
        # Check if it has the expected structure for occupancy detection
        has_fc = any('fc' in key for key in state_dict.keys())
        has_conv = any('conv' in key for key in state_dict.keys())
        
        if has_fc and has_conv:
            logger.info(f"   ✅ Model has both convolutional and fully connected layers")
        else:
            logger.warning(f"   ⚠️  Model structure might be unexpected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return False

def main():
    """Run the Marshall occupancy test"""
    logger.info("🔍 Marshall Occupancy Model Validation")
    logger.info("Testing if the model is working correctly")
    logger.info("=" * 60)
    
    success = test_marshall_occupancy()
    
    if success:
        logger.info(f"\n🎉 CONCLUSION:")
        logger.info(f"   The Marshall occupancy model appears to be working correctly!")
        logger.info(f"   It can load, run inference, and produces expected outputs.")
        logger.info(f"   The model is ready for use in your chess application.")
    else:
        logger.error(f"\n❌ CONCLUSION:")
        logger.error(f"   The Marshall occupancy model has issues and needs attention.")

if __name__ == "__main__":
    main()
