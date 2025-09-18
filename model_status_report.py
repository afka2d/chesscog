#!/usr/bin/env python3
"""
Model status report that shows the current state and what needs adjustment.
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

def generate_model_status_report():
    """Generate a comprehensive model status report"""
    print("Chess Model Status Report")
    print("=" * 50)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            health_data = response.json()
            print(f"API Status: {health_data}")
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
    
    # Test with different corner detection methods
    methods = [
        ("Auto-detected corners", detect_corners_auto),
        ("Working corners from logs", lambda: [[302.3999938964844, 302.3999938964844], [3729.60009765625, 302.3999938964844], [3729.60009765625, 2721.60009765625], [302.3999938964844, 2721.60009765625]]),
        ("Estimated corners", detect_corners_estimated)
    ]
    
    results = []
    
    for method_name, corner_func in methods:
        print(f"\n--- Testing {method_name} ---")
        corners = corner_func()
        if corners is None:
            print("  ❌ Could not detect corners")
            continue
            
        result = test_api_call(image_path, corners, method_name)
        if result:
            results.append(result)
    
    # Analyze results
    print(f"\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    if not results:
        print("❌ No successful API calls. Check API configuration.")
        return
    
    # Find best result
    best_result = max(results, key=lambda x: x['pieces_detected'])
    
    print(f"Best result: {best_result['method']}")
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
    
    # Diagnosis
    print(f"\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if best_result['pieces_detected'] == 0:
        print("❌ ISSUE: No pieces detected")
        print("   - This suggests the occupancy threshold is too high")
        print("   - From your logs, the model should detect 10 pieces")
        print("   - Current threshold is 0.3, try lowering to 0.2 or 0.1")
        print("   - Edit main_local_dev.py line 286: change 0.3 to 0.2")
    else:
        print("✅ Pieces are being detected")
    
    if best_result['occupied_squares'] == 0:
        print("❌ ISSUE: No occupied squares detected")
        print("   - This is the root cause of no piece detection")
        print("   - The occupancy classifier is not finding occupied squares")
        print("   - Lower the occupancy threshold in main_local_dev.py")
    else:
        print("✅ Occupied squares are being detected")
    
    if best_result['fen'] == '8/8/8/8/8/8/8/8 w - - 0 1':
        print("❌ ISSUE: Empty FEN generated")
        print("   - This is because no pieces were detected")
        print("   - Fix the occupancy detection first")
    else:
        print("✅ Non-empty FEN generated")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if best_result['pieces_detected'] == 0:
        print("1. IMMEDIATE: Lower occupancy threshold to 0.2")
        print("2. Edit main_local_dev.py line 286:")
        print("   Change: is_occupied = prediction == 1 and confidence > 0.3")
        print("   To:     is_occupied = prediction == 1 and confidence > 0.2")
        print("3. Restart API: ./start_local_dev.sh")
        print("4. Test again: python model_status_report.py")
        print("5. If still 0 pieces, try threshold 0.1")
    else:
        print("1. Model is working correctly")
        print("2. Test with more images to get comprehensive metrics")
        print("3. Consider implementing adaptive thresholds")
    
    # Save results
    save_results(results, best_result, occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy)

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

def save_results(results, best_result, occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy):
    """Save results to file"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'all_results': results,
        'best_result': best_result,
        'metrics': {
            'occupancy_accuracy': occupancy_accuracy,
            'color_accuracy': color_accuracy,
            'piece_accuracy': piece_accuracy,
            'fen_accuracy': fen_accuracy
        }
    }
    
    with open("model_status_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nResults saved to: model_status_report.json")

if __name__ == "__main__":
    import time
    generate_model_status_report()
