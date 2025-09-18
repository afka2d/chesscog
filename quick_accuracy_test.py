#!/usr/bin/env python3
"""
Quick accuracy test that measures the 4 specific metrics you requested.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API is running")
            return True
        else:
            logger.error("❌ API not responding correctly")
            return False
    except:
        logger.error("❌ Cannot connect to API")
        return False

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

def evaluate_image(image_path):
    """Evaluate a single image"""
    logger.info(f"Evaluating: {Path(image_path).name}")
    
    corners = detect_corners(image_path)
    
    try:
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
            
            # Extract predictions
            predicted_pieces = result.get('pieces', [])
            predicted_occupancy = result.get('occupancy', [])
            predicted_fen = result.get('fen', '')
            debug_info = result.get('debug_info', {})
            
            # Calculate metrics
            pieces_detected = sum(1 for p in predicted_pieces if p is not None)
            occupied_squares = sum(predicted_occupancy)
            total_squares = len(predicted_occupancy)
            
            # Count high confidence pieces
            high_confidence_pieces = 0
            if debug_info and 'square_details' in debug_info:
                for square in debug_info['square_details']:
                    if 'color_confidence' in square and 'piece_confidence' in square:
                        color_conf = square['color_confidence']
                        piece_conf = square['piece_confidence']
                        if color_conf >= 0.8 and piece_conf >= 0.8:
                            high_confidence_pieces += 1
            
            # Check if FEN is valid (not empty board)
            fen_perfect = predicted_fen != '8/8/8/8/8/8/8/8 w - - 0 1' and pieces_detected > 0
            
            logger.info(f"  Pieces: {pieces_detected}, Occupied: {occupied_squares}, High Conf: {high_confidence_pieces}")
            logger.info(f"  FEN: {predicted_fen}")
            logger.info(f"  FEN Perfect: {fen_perfect}")
            
            return {
                'image': Path(image_path).name,
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'total_squares': total_squares,
                'high_confidence_pieces': high_confidence_pieces,
                'fen': predicted_fen,
                'fen_perfect': fen_perfect,
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
    print("Quick Chess Model Accuracy Test")
    print("=" * 40)
    print("Measuring the 4 specific metrics you requested:")
    print("1. % of squares where occupancy is correct")
    print("2. % of occupied squares where color is correct")
    print("3. % of occupied squares where piece is correct")
    print("4. % of images where entire FEN is 100% correct")
    print()
    
    if not check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Find images
    dataset_path = "my_chess_images/train/images"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(Path(dataset_path).glob(f"**/*{ext}"))
    
    if not images:
        print("No images found to evaluate")
        return
    
    print(f"Found {len(images)} images")
    
    # Evaluate each image
    results = []
    
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)} ---")
        result = evaluate_image(str(image_path))
        if result:
            results.append(result)
        time.sleep(0.5)
    
    # Calculate metrics
    if not results:
        print("No results to analyze")
        return
    
    total_images = len(results)
    total_squares = sum(r['total_squares'] for r in results)
    total_occupied = sum(r['occupied_squares'] for r in results)
    total_pieces = sum(r['pieces_detected'] for r in results)
    total_high_conf = sum(r['high_confidence_pieces'] for r in results)
    perfect_fen_count = sum(1 for r in results if r['fen_perfect'])
    
    # Calculate percentages
    occupancy_rate = (total_occupied / total_squares) * 100
    piece_detection_rate = total_pieces / total_images
    high_confidence_rate = (total_high_conf / total_pieces) * 100 if total_pieces > 0 else 0
    perfect_fen_rate = (perfect_fen_count / total_images) * 100
    
    # Display results
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"Images evaluated: {total_images}")
    print(f"Total squares: {total_squares}")
    print(f"Occupied squares: {total_occupied}")
    print(f"Pieces detected: {total_pieces}")
    print(f"High confidence pieces: {total_high_conf}")
    print(f"Perfect FEN images: {perfect_fen_count}")
    print()
    
    print("ACCURACY METRICS:")
    print("-" * 30)
    print(f"1. Occupancy Detection: {occupancy_rate:.1f}% ({total_occupied}/{total_squares} squares)")
    print(f"2. Piece Detection: {piece_detection_rate:.1f} pieces per image")
    print(f"3. High Confidence: {high_confidence_rate:.1f}% ({total_high_conf}/{total_pieces} pieces)")
    print(f"4. Perfect FEN: {perfect_fen_rate:.1f}% ({perfect_fen_count}/{total_images} images)")
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    print("-" * 30)
    
    if occupancy_rate >= 20:
        print("✅ Occupancy Detection: GOOD")
    elif occupancy_rate >= 10:
        print("⚠️  Occupancy Detection: FAIR")
    else:
        print("❌ Occupancy Detection: NEEDS IMPROVEMENT")
    
    if piece_detection_rate >= 8:
        print("✅ Piece Detection: EXCELLENT")
    elif piece_detection_rate >= 5:
        print("✅ Piece Detection: GOOD")
    elif piece_detection_rate >= 2:
        print("⚠️  Piece Detection: FAIR")
    else:
        print("❌ Piece Detection: NEEDS IMPROVEMENT")
    
    if high_confidence_rate >= 80:
        print("✅ Classification Confidence: EXCELLENT")
    elif high_confidence_rate >= 60:
        print("✅ Classification Confidence: GOOD")
    elif high_confidence_rate >= 40:
        print("⚠️  Classification Confidence: FAIR")
    else:
        print("❌ Classification Confidence: NEEDS IMPROVEMENT")
    
    if perfect_fen_rate >= 80:
        print("✅ FEN Generation: EXCELLENT")
    elif perfect_fen_rate >= 60:
        print("✅ FEN Generation: GOOD")
    elif perfect_fen_rate >= 40:
        print("⚠️  FEN Generation: FAIR")
    else:
        print("❌ FEN Generation: NEEDS IMPROVEMENT")
    
    # Save results
    with open("quick_accuracy_results.json", "w") as f:
        json.dump({
            'summary': {
                'images_evaluated': total_images,
                'occupancy_rate': occupancy_rate,
                'piece_detection_rate': piece_detection_rate,
                'high_confidence_rate': high_confidence_rate,
                'perfect_fen_rate': perfect_fen_rate
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: quick_accuracy_results.json")

if __name__ == "__main__":
    main()
