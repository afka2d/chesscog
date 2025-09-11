#!/usr/bin/env python3
"""
Direct test of the production API using the exact same parameters that worked locally.
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_production_api():
    """Test the production API directly"""
    logger.info("🧪 Testing Production API Directly")
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
    
    # Test with a simple image and known corners
    logger.info("🔍 Testing with simple image...")
    
    # Use the exact corners that worked in our debug
    corners = [[324.0, 324.0], [2916.0, 324.0], [2916.0, 5436.0], [324.0, 5436.0]]
    
    try:
        # Create a simple test image (white square)
        import numpy as np
        import cv2
        from PIL import Image
        import io
        
        # Create a simple test image
        test_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # White image
        
        # Convert to bytes
        _, img_bytes = cv2.imencode('.jpg', test_image)
        img_bytes = img_bytes.tobytes()
        
        files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
        data = {'corners': json.dumps(corners)}
        
        response = requests.post(
            "https://api.chesspositionscanner.store/recognize_chess_position_with_corners",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ API call successful!")
            logger.info(f"📊 Results:")
            logger.info(f"   FEN: {result.get('fen', 'N/A')}")
            logger.info(f"   Pieces detected: {sum(1 for p in result.get('pieces', []) if p is not None)}")
            logger.info(f"   Occupancy: {sum(result.get('occupancy', []))} occupied squares")
            logger.info(f"   Success: {result.get('success', 'N/A')}")
            
            return True
        else:
            logger.error(f"❌ API call failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_production_api()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed. Check the logs above.")
