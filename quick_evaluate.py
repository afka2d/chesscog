#!/usr/bin/env python3
"""
Quick evaluation script for model accuracy testing.
This works with your local development API on port 8001.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api():
    """Check if local API is running"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Local API is running")
            return True
        else:
            logger.error("❌ Local API not responding")
            return False
    except:
        logger.error("❌ Cannot connect to local API")
        return False

def detect_corners(image_path):
    """Detect chessboard corners"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try to find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    
    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Convert to the format expected by the API
        corners_2d = corners.reshape(-1, 2)
        
        # Find the 4 outer corners
        top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
        top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
        bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
        bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
        
        return [top_left, top_right, bottom_right, bottom_left]
    else:
        # Fallback: estimate corners
        h, w = img.shape[:2]
        margin = min(h, w) * 0.1
        
        return [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]

def evaluate_image(image_path):
    """Evaluate a single image"""
    logger.info(f"Evaluating: {Path(image_path).name}")
    
    try:
        # Detect corners
        corners = detect_corners(image_path)
        
        # Call API with debug mode
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'debug': 'true'
            }
            
            response = requests.post(
                "http://localhost:8001/recognize_chess_position_with_corners",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract key metrics
            pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
            occupied_squares = sum(result.get('occupancy', []))
            fen = result.get('fen', '')
            
            # Get debug info
            debug_info = result.get('debug_info', {})
            processing_time = debug_info.get('processing_time', 0)
            
            logger.info(f"  Pieces detected: {pieces_detected}")
            logger.info(f"  Occupied squares: {occupied_squares}")
            logger.info(f"  Processing time: {processing_time:.3f}s")
            logger.info(f"  FEN: {fen}")
            
            return {
                'image': Path(image_path).name,
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'processing_time': processing_time,
                'fen': fen,
                'success': result.get('success', False)
            }
        else:
            logger.error(f"API call failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error evaluating {image_path}: {e}")
        return None

def main():
    """Main evaluation function"""
    print("Chess Model Quick Evaluation")
    print("=" * 40)
    
    # Check API
    if not check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Find images to evaluate
    dataset_path = Path("my_chess_images/train/images")
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not images:
        logger.error("No images found to evaluate")
        return
    
    logger.info(f"Found {len(images)} images to evaluate")
    
    # Limit to first 5 images for quick test
    images = images[:5]
    
    results = []
    
    # Evaluate each image
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)} ---")
        result = evaluate_image(str(image_path))
        if result:
            results.append(result)
        
        # Small delay
        time.sleep(1)
    
    # Summary
    if results:
        print("\n" + "=" * 40)
        print("EVALUATION SUMMARY")
        print("=" * 40)
        
        total_pieces = sum(r['pieces_detected'] for r in results)
        total_occupied = sum(r['occupied_squares'] for r in results)
        avg_time = np.mean([r['processing_time'] for r in results])
        
        print(f"Images processed: {len(results)}")
        print(f"Total pieces detected: {total_pieces}")
        print(f"Total occupied squares: {total_occupied}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        # Save results
        with open("quick_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: quick_evaluation_results.json")
    else:
        print("No results to summarize")

if __name__ == "__main__":
    main()
