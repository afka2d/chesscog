#!/usr/bin/env python3
"""
Test script for the local development API.
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

def test_local_dev_api():
    """Test the local development API"""
    logger.info("🧪 Testing Local Development API")
    logger.info("=" * 50)
    
    # Test health check
    logger.info("🔍 Testing health check...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Health check passed: {health_data}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False
    
    # Test debug info endpoint
    logger.info("🔍 Testing debug info endpoint...")
    try:
        response = requests.get("http://localhost:8001/debug/info", timeout=10)
        if response.status_code == 200:
            debug_data = response.json()
            logger.info(f"✅ Debug info: {debug_data}")
        else:
            logger.error(f"❌ Debug info failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Debug info failed: {e}")
    
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
            data = {
                'corners': json.dumps(corners),
                'debug': 'true'  # Enable debug mode
            }
            
            response = requests.post(
                "http://localhost:8001/recognize_chess_position_with_corners",
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
            
            # Show debug info if available
            if 'debug_info' in result:
                debug_info = result['debug_info']
                logger.info(f"🔧 Debug Info:")
                logger.info(f"   Squares processed: {debug_info.get('squares_processed', 0)}")
                logger.info(f"   Occupied squares: {debug_info.get('occupied_squares', 0)}")
                logger.info(f"   Pieces detected: {debug_info.get('pieces_detected', 0)}")
                logger.info(f"   Processing time: {debug_info.get('processing_time', 0):.3f}s")
                
                # Show confidence scores for occupied squares
                confidence_scores = debug_info.get('confidence_scores', [])
                if confidence_scores:
                    logger.info(f"   Confidence scores for occupied squares:")
                    for score in confidence_scores[:5]:  # Show first 5
                        logger.info(f"     {score['square']}: occ={score['occupancy_confidence']:.3f}, color={score['color_confidence']:.3f}, piece={score['piece_confidence']:.3f}")
            
            return True
        else:
            logger.error(f"❌ Chess position recognition failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Chess position recognition failed: {e}")
        return False

if __name__ == "__main__":
    success = test_local_dev_api()
    if success:
        logger.info("🎉 Local Development API test completed successfully!")
        logger.info("✅ You can now work on improvements without affecting production!")
    else:
        logger.error("💥 Test failed. Check the logs above.")
