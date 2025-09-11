#!/usr/bin/env python3
"""
Final test of the production API to confirm it's working correctly.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_production_api():
    """Test the production API with real chess image"""
    logger.info("🎯 Final Production API Test")
    logger.info("=" * 50)
    
    # Test health check
    logger.info("🔍 Testing health check...")
    try:
        response = requests.get("https://api.chesspositionscanner.store/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Health check passed: {health_data}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False
    
    # Test with real chess image
    logger.info("🔍 Testing with real chess image...")
    
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    if not Path(image_path).exists():
        logger.error(f"❌ Test image not found: {image_path}")
        return False
    
    # Use detected corners
    corners = [[324.0, 324.0], [2916.0, 324.0], [2916.0, 5436.0], [324.0, 5436.0]]
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'corners': json.dumps(corners)}
            
            response = requests.post(
                "https://api.chesspositionscanner.store/recognize_chess_position_with_corners",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Chess position recognition successful!")
            logger.info(f"📊 Results:")
            logger.info(f"   FEN: {result.get('fen', 'N/A')}")
            logger.info(f"   Pieces detected: {sum(1 for p in result.get('pieces', []) if p is not None)}")
            logger.info(f"   Occupancy: {sum(result.get('occupancy', []))} occupied squares")
            logger.info(f"   Success: {result.get('success', 'N/A')}")
            
            # Show response format
            logger.info("📋 Response format:")
            logger.info(f"   - FEN: {type(result.get('fen'))}")
            logger.info(f"   - Pieces: {type(result.get('pieces'))} (length: {len(result.get('pieces', []))})")
            logger.info(f"   - Occupancy: {type(result.get('occupancy'))} (length: {len(result.get('occupancy', []))})")
            logger.info(f"   - Success: {type(result.get('success'))}")
            
            return True
        else:
            logger.error(f"❌ Chess position recognition failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Chess position recognition failed: {e}")
        return False

if __name__ == "__main__":
    success = test_production_api()
    if success:
        logger.info("🎉 Production API is working correctly!")
        logger.info("✅ Your app should now work with the production API!")
    else:
        logger.error("💥 Test failed. Check the logs above.")
