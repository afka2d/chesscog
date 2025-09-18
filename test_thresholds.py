#!/usr/bin/env python3
"""
Test different occupancy thresholds to find optimal settings.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_corners(image_path):
    """Detect chessboard corners"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners_2d = corners.reshape(-1, 2)
        
        top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
        top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
        bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
        bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
        
        return [top_left, top_right, bottom_right, bottom_left]
    else:
        h, w = img.shape[:2]
        margin = min(h, w) * 0.1
        return [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]

def test_image_with_threshold(image_path, threshold):
    """Test an image with a specific threshold by modifying the API call"""
    logger.info(f"Testing {Path(image_path).name} with threshold {threshold}")
    
    corners = detect_corners(image_path)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'corners': json.dumps(corners),
                'debug': 'true',
                'threshold': str(threshold)  # Pass threshold as parameter
            }
            
            response = requests.post(
                "http://localhost:8001/recognize_chess_position_with_corners",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
            occupied_squares = sum(result.get('occupancy', []))
            fen = result.get('fen', '')
            
            return {
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'fen': fen,
                'success': result.get('success', False)
            }
        else:
            logger.error(f"API call failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error testing {image_path}: {e}")
        return None

def main():
    """Test different thresholds"""
    print("Testing Different Occupancy Thresholds")
    print("=" * 40)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ Local API is not running!")
            print("Please start it with: ./start_local_dev.sh")
            return
    except:
        print("❌ Cannot connect to local API!")
        print("Please start it with: ./start_local_dev.sh")
        return
    
    # Find test images
    dataset_path = Path("my_chess_images/train/images")
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not images:
        print("❌ No images found to test")
        return
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"Testing {len(images)} images with {len(thresholds)} thresholds...")
    
    results = {}
    
    for image_path in images:
        image_name = Path(image_path).name
        print(f"\n--- Testing {image_name} ---")
        
        image_results = {}
        
        for threshold in thresholds:
            result = test_image_with_threshold(str(image_path), threshold)
            if result:
                image_results[threshold] = result
                print(f"  Threshold {threshold}: {result['pieces_detected']} pieces, {result['occupied_squares']} occupied")
            else:
                print(f"  Threshold {threshold}: Failed")
        
        results[image_name] = image_results
    
    # Summary
    print("\n" + "=" * 40)
    print("THRESHOLD TEST SUMMARY")
    print("=" * 40)
    
    for image_name, image_results in results.items():
        print(f"\n{image_name}:")
        for threshold, result in image_results.items():
            print(f"  {threshold}: {result['pieces_detected']} pieces, {result['occupied_squares']} occupied")
    
    # Find optimal threshold
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS")
    print("=" * 40)
    
    # Analyze results to find best threshold
    threshold_performance = {}
    
    for threshold in thresholds:
        total_pieces = 0
        total_occupied = 0
        image_count = 0
        
        for image_name, image_results in results.items():
            if threshold in image_results:
                total_pieces += image_results[threshold]['pieces_detected']
                total_occupied += image_results[threshold]['occupied_squares']
                image_count += 1
        
        if image_count > 0:
            avg_pieces = total_pieces / image_count
            avg_occupied = total_occupied / image_count
            threshold_performance[threshold] = {
                'avg_pieces': avg_pieces,
                'avg_occupied': avg_occupied,
                'total_pieces': total_pieces
            }
    
    # Find threshold with most pieces detected
    best_threshold = max(threshold_performance.keys(), key=lambda t: threshold_performance[t]['total_pieces'])
    
    print(f"Best threshold based on total pieces detected: {best_threshold}")
    print(f"  Total pieces: {threshold_performance[best_threshold]['total_pieces']}")
    print(f"  Average pieces per image: {threshold_performance[best_threshold]['avg_pieces']:.1f}")
    print(f"  Average occupied squares per image: {threshold_performance[best_threshold]['avg_occupied']:.1f}")
    
    # Save results
    with open("threshold_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: threshold_test_results.json")

if __name__ == "__main__":
    main()
