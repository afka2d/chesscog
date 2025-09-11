#!/usr/bin/env python3
"""
Simple test of the production API with the working commit.
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

def detect_chessboard_corners(image_path):
    """Detect chessboard corners using OpenCV"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try to find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    
    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Convert to the format expected by the API
        # We need 4 corners, so we'll take the outer corners
        h, w = gray.shape
        corner_points = []
        
        # Find the 4 outer corners
        corners_2d = corners.reshape(-1, 2)
        
        # Top-left: minimum x+y
        top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
        
        # Top-right: maximum x-y
        top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
        
        # Bottom-right: maximum x+y
        bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
        
        # Bottom-left: minimum x-y
        bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
        
        corner_points = [top_left, top_right, bottom_right, bottom_left]
        
        return corner_points
    else:
        # Fallback: estimate corners based on image dimensions
        h, w = img.shape[:2]
        margin = min(h, w) * 0.1
        
        return [
            [margin, margin],  # Top-left
            [w - margin, margin],  # Top-right
            [w - margin, h - margin],  # Bottom-right
            [margin, h - margin]  # Bottom-left
        ]

def test_production_api():
    """Test the production API with real chess image"""
    logger.info("🧪 Testing Production API with Working Commit")
    logger.info("=" * 60)
    
    # Test health check
    logger.info("🔍 Testing health check endpoint...")
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
    
    # Test with a real chess image
    logger.info("🔍 Testing chess position recognition...")
    
    # Use a real chess image
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    if not Path(image_path).exists():
        logger.error(f"❌ Test image not found: {image_path}")
        return False
    
    # Detect corners automatically
    logger.info("🔍 Detecting chessboard corners...")
    corners = detect_chessboard_corners(image_path)
    logger.info(f"Detected corners: {corners}")
    
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
            
            # Show some pieces
            pieces = result.get('pieces', [])
            if pieces:
                logger.info("   Sample pieces:")
                for i, piece in enumerate(pieces):
                    if piece is not None:
                        rank = 8 - (i // 8)
                        file = chr(ord('a') + (i % 8))
                        logger.info(f"     {file}{rank}: {piece}")
            
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
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed. Check the logs above.")
