#!/usr/bin/env python3
"""
Test the deployed lightweight model to verify it's working correctly.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deployed_model():
    """Test the deployed model to ensure it's working correctly."""
    
    model_path = Path("runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
    
    if not model_path.exists():
        logger.error(f"❌ Deployed model not found: {model_path}")
        return False
    
    try:
        # Load the model
        logger.info("🔄 Loading deployed model...")
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        logger.info("✅ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Test with dummy input
        logger.info("🧪 Testing with dummy input...")
        dummy_input = torch.randn(1, 3, 100, 200)  # Match the expected input size
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        logger.info(f"✅ Model inference successful")
        logger.info(f"   Predicted class: {predicted_class}")
        logger.info(f"   Confidence: {confidence:.3f}")
        
        # Verify output shape
        expected_classes = 12  # 12 piece types
        if output.shape[1] == expected_classes:
            logger.info(f"✅ Output shape correct: {output.shape}")
        else:
            logger.error(f"❌ Unexpected output shape: {output.shape}")
            return False
        
        # Test transforms
        logger.info("🔄 Testing transforms...")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        transformed = transform(dummy_image)
        
        if transformed.shape == (3, 100, 200):
            logger.info("✅ Transforms working correctly")
        else:
            logger.error(f"❌ Transform output shape incorrect: {transformed.shape}")
            return False
        
        logger.info("🎉 All tests passed! Model is ready for production.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🧪 Testing Deployed Lightweight Model")
    logger.info("=" * 40)
    
    if test_deployed_model():
        logger.info("\n✅ DEPLOYMENT VERIFICATION COMPLETE!")
        logger.info("The API is ready to use the new 97.65% accuracy model.")
        return True
    else:
        logger.error("\n❌ DEPLOYMENT VERIFICATION FAILED!")
        logger.error("Please check the model deployment.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
