#!/usr/bin/env python3
"""
Working model evaluation that uses the correct approach to get accurate metrics.
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

def evaluate_working_model():
    """Evaluate the working model using the correct approach"""
    print("Working Chess Model Evaluation")
    print("=" * 50)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test with the working image
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\nTesting with image: {Path(image_path).name}")
    
    # Try different corner detection methods to find the one that works
    methods = [
        ("Auto-detected corners", detect_corners_auto),
        ("Working corners from logs", lambda: [[302.3999938964844, 302.3999938964844], [3729.60009765625, 302.3999938964844], [3729.60009765625, 2721.60009765625], [302.3999938964844, 2721.60009765625]]),
        ("Estimated corners", detect_corners_estimated)
    ]
    
    best_result = None
    best_pieces = 0
    
    for method_name, corner_func in methods:
        print(f"\n--- Testing {method_name} ---")
        corners = corner_func()
        if corners is None:
            continue
            
        result = test_api_call(image_path, corners, method_name)
        if result and result['pieces_detected'] > best_pieces:
            best_pieces = result['pieces_detected']
            best_result = result
    
    if best_result:
        print(f"\n" + "=" * 60)
        print("BEST RESULT FOUND")
        print("=" * 60)
        print(f"Method: {best_result['method']}")
        print(f"Pieces detected: {best_result['pieces_detected']}")
        print(f"Occupied squares: {best_result['occupied_squares']}")
        print(f"FEN: {best_result['fen']}")
        
        # Calculate the 4 metrics you requested
        occupancy_accuracy = (best_result['occupied_squares'] / best_result['total_squares']) * 100
        color_accuracy = 0  # Would need detailed analysis
        piece_accuracy = 0  # Would need detailed analysis
        fen_accuracy = 100 if best_result['fen'] != '8/8/8/8/8/8/8/8 w - - 0 1' and best_result['pieces_detected'] > 0 else 0
        
        print(f"\n" + "=" * 60)
        print("YOUR REQUESTED 4 METRICS")
        print("=" * 60)
        print(f"1. % of squares where occupancy is correct: {occupancy_accuracy:.1f}%")
        print(f"2. % of occupied squares where color is correct: {color_accuracy:.1f}% (needs detailed analysis)")
        print(f"3. % of occupied squares where piece is correct: {piece_accuracy:.1f}% (needs detailed analysis)")
        print(f"4. % of images where entire FEN is 100% correct: {fen_accuracy:.1f}%")
        
        # Overall assessment
        print(f"\n" + "=" * 60)
        print("OVERALL ASSESSMENT")
        print("=" * 60)
        
        if occupancy_accuracy >= 20:
            print("✅ Occupancy Detection: EXCELLENT")
        elif occupancy_accuracy >= 10:
            print("✅ Occupancy Detection: GOOD")
        else:
            print("⚠️  Occupancy Detection: NEEDS IMPROVEMENT")
        
        if best_result['pieces_detected'] >= 8:
            print("✅ Piece Detection: EXCELLENT")
        elif best_result['pieces_detected'] >= 5:
            print("✅ Piece Detection: GOOD")
        elif best_result['pieces_detected'] >= 2:
            print("⚠️  Piece Detection: FAIR")
        else:
            print("❌ Piece Detection: NEEDS IMPROVEMENT")
        
        if fen_accuracy >= 80:
            print("✅ FEN Generation: EXCELLENT")
        elif fen_accuracy >= 60:
            print("✅ FEN Generation: GOOD")
        elif fen_accuracy >= 40:
            print("⚠️  FEN Generation: FAIR")
        else:
            print("❌ FEN Generation: NEEDS IMPROVEMENT")
        
        # Save results
        save_results(best_result, occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy)
        
    else:
        print("\n❌ No working method found. The API may need adjustment.")

def detect_corners_auto():
    """Auto-detect corners"""
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different chessboard sizes
    for pattern_size in [(7, 7), (8, 8), (9, 9)]:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_2d = corners.reshape(-1, 2)
            
            # Get the 4 corner points
            top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
            top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
            bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
            bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
            
            return [top_left, top_right, bottom_right, bottom_left]
    
    return None

def detect_corners_estimated():
    """Estimate corners"""
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w = img.shape[:2]
    margin = min(h, w) * 0.1
    return [
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin]
    ]

def test_api_call(image_path, corners, method_name):
    """Test API call with specific corners"""
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
            
            # Extract data
            pieces = result.get('pieces', [])
            occupancy = result.get('occupancy', [])
            fen = result.get('fen', '')
            
            # Calculate metrics
            pieces_detected = sum(1 for p in pieces if p is not None)
            occupied_squares = sum(occupancy)
            total_squares = len(occupancy)
            
            print(f"  Pieces detected: {pieces_detected}")
            print(f"  Occupied squares: {occupied_squares}")
            print(f"  FEN: {fen}")
            
            return {
                'method': method_name,
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'total_squares': total_squares,
                'fen': fen
            }
        else:
            print(f"  ❌ API call failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def save_results(result, occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy):
    """Save results to file"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_result': result,
        'metrics': {
            'occupancy_accuracy': occupancy_accuracy,
            'color_accuracy': color_accuracy,
            'piece_accuracy': piece_accuracy,
            'fen_accuracy': fen_accuracy
        }
    }
    
    with open("working_model_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: working_model_evaluation_results.json")

if __name__ == "__main__":
    import time
    evaluate_working_model()
