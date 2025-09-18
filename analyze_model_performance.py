#!/usr/bin/env python3
"""
Analyze model performance by testing different thresholds and confidence levels.
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

def test_api_endpoints():
    """Test different API endpoints and configurations"""
    print("Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8001/health")
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
    except Exception as e:
        print(f"Health endpoint error: {e}")
    
    # Test debug info endpoint
    try:
        response = requests.get("http://localhost:8001/debug/info")
        print(f"Debug info endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Debug info: {response.json()}")
    except Exception as e:
        print(f"Debug info endpoint error: {e}")

def analyze_image_with_different_settings(image_path):
    """Analyze an image with different API settings"""
    logger.info(f"Analyzing: {Path(image_path).name}")
    
    corners = detect_corners(image_path)
    
    # Test with different debug settings
    test_cases = [
        {"debug": "true", "name": "Debug enabled"},
        {"debug": "false", "name": "Debug disabled"},
    ]
    
    results = {}
    
    for test_case in test_cases:
        logger.info(f"  Testing: {test_case['name']}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'corners': json.dumps(corners),
                    'debug': test_case['debug']
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
                
                # Get debug info if available
                debug_info = result.get('debug_info', {})
                processing_time = debug_info.get('processing_time', 0)
                
                results[test_case['name']] = {
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'processing_time': processing_time,
                    'fen': fen,
                    'debug_info': debug_info
                }
                
                logger.info(f"    Pieces: {pieces_detected}, Occupied: {occupied_squares}, Time: {processing_time:.3f}s")
                
            else:
                logger.error(f"    API call failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"    Error: {e}")
    
    return results

def main():
    """Main analysis function"""
    print("Chess Model Performance Analysis")
    print("=" * 40)
    
    # Check API
    if not check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Test API endpoints
    test_api_endpoints()
    
    # Find images to analyze
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
        logger.error("No images found to analyze")
        return
    
    logger.info(f"Found {len(images)} images to analyze")
    
    # Analyze each image
    all_results = {}
    
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)}: {Path(image_path).name} ---")
        results = analyze_image_with_different_settings(str(image_path))
        all_results[str(image_path)] = results
        
        # Small delay
        time.sleep(1)
    
    # Summary analysis
    print("\n" + "=" * 40)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 40)
    
    for image_path, results in all_results.items():
        print(f"\nImage: {Path(image_path).name}")
        for test_name, result in results.items():
            print(f"  {test_name}:")
            print(f"    Pieces: {result['pieces_detected']}")
            print(f"    Occupied: {result['occupied_squares']}")
            print(f"    Time: {result['processing_time']:.3f}s")
            print(f"    FEN: {result['fen']}")
    
    # Save results
    with open("model_performance_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: model_performance_analysis.json")

if __name__ == "__main__":
    main()
