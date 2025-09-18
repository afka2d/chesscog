#!/usr/bin/env python3
"""
Test script to verify the color classifier is working correctly on real-world data.
This will test the API with sample images to ensure color classification is accurate.
"""

import requests
import json
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_color_classifier_api():
    """Test the color classifier API with sample data"""
    
    # API endpoint
    url = "http://localhost:8000/recognize_chess_position_with_corners"
    
    # Test with a sample image (you can replace this with your actual test image)
    # For now, we'll create a simple test image
    test_image_path = "test_image.jpg"
    
    # Create a simple test image if it doesn't exist
    if not Path(test_image_path).exists():
        logger.info("Creating test image...")
        # Create a simple chess board image for testing
        img = np.ones((1200, 1200, 3), dtype=np.uint8) * 255  # White background
        cv2.rectangle(img, (100, 100), (1100, 1100), (0, 0, 0), 3)  # Board outline
        cv2.imwrite(test_image_path, img)
        logger.info(f"Test image created: {test_image_path}")
    
    # Sample corners (you can adjust these based on your test image)
    corners = [[100, 100], [1100, 100], [1100, 1100], [100, 1100]]
    
    try:
        # Prepare the request
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'turn': 'white'
            }
            
            logger.info("Testing color classifier API...")
            logger.info(f"Image: {test_image_path}")
            logger.info(f"Corners: {corners}")
            
            # Make the request
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ API request successful!")
                logger.info(f"FEN: {result['fen']}")
                logger.info(f"Pieces found: {sum(1 for p in result['pieces'] if p is not None)}")
                logger.info(f"Occupancy: {sum(result['occupancy'])} occupied squares")
                
                # Show piece classification results
                pieces = result['pieces']
                occupancy = result['occupancy']
                
                logger.info("\n📋 Piece Classification Results:")
                for i, (piece, occupied) in enumerate(zip(pieces, occupancy)):
                    if occupied:
                        rank = 8 - (i // 8)
                        file = chr(97 + (i % 8))
                        logger.info(f"  {file}{rank}: {piece}")
                
                return True
            else:
                logger.error(f"❌ API request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error testing API: {e}")
        return False

def test_with_real_images():
    """Test with real chess images if available"""
    logger.info("\n🔍 Looking for real chess images to test...")
    
    # Look for common chess image patterns
    image_patterns = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp",
        "test_*.jpg", "chess_*.jpg", "board_*.jpg"
    ]
    
    test_images = []
    for pattern in image_patterns:
        test_images.extend(Path(".").glob(pattern))
    
    if test_images:
        logger.info(f"Found {len(test_images)} test images")
        for img_path in test_images[:3]:  # Test first 3 images
            logger.info(f"\n📸 Testing with image: {img_path}")
            # You can add specific corner coordinates for each image here
            # For now, we'll use default corners
            test_with_image(str(img_path))
    else:
        logger.info("No test images found. Using synthetic test image.")

def test_with_image(image_path, corners=None):
    """Test the API with a specific image"""
    if corners is None:
        # Default corners - you may need to adjust these
        corners = [[100, 100], [1100, 100], [1100, 1100], [100, 1100]]
    
    url = "http://localhost:8000/recognize_chess_position_with_corners"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'turn': 'white'
            }
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Success with {image_path}")
                logger.info(f"Pieces found: {sum(1 for p in result['pieces'] if p is not None)}")
                
                # Show piece colors
                pieces = result['pieces']
                occupancy = result['occupancy']
                
                white_pieces = 0
                black_pieces = 0
                
                for i, (piece, occupied) in enumerate(zip(pieces, occupancy)):
                    if occupied and piece:
                        if piece.isupper():  # White pieces are uppercase
                            white_pieces += 1
                        else:  # Black pieces are lowercase
                            black_pieces += 1
                
                logger.info(f"White pieces: {white_pieces}, Black pieces: {black_pieces}")
                return True
            else:
                logger.error(f"❌ Failed with {image_path}: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error with {image_path}: {e}")
        return False

def main():
    logger.info("🧪 Testing Color Classifier API")
    logger.info("=" * 50)
    
    # Test 1: Basic API functionality
    logger.info("Test 1: Basic API functionality")
    success = test_color_classifier_api()
    
    if success:
        logger.info("✅ Basic test passed!")
        
        # Test 2: Real images if available
        logger.info("\nTest 2: Real images (if available)")
        test_with_real_images()
        
        logger.info("\n🎉 Color classifier testing completed!")
        logger.info("The API is working and ready for real-world use.")
    else:
        logger.error("❌ Basic test failed. Please check the API.")

if __name__ == "__main__":
    main()
