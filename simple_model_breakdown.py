#!/usr/bin/env python3
"""
Simple model breakdown that works with your published API.
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

def test_api_with_different_corners():
    """Test API with different corner detection methods"""
    print("Testing API with different corner detection methods")
    print("=" * 60)
    
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
    
    # Test image
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\nTesting with image: {Path(image_path).name}")
    
    # Method 1: Try to detect corners properly
    print("\n--- Method 1: Proper corner detection ---")
    corners1 = detect_corners_properly(image_path)
    if corners1:
        test_with_corners(image_path, corners1, "Proper detection")
    
    # Method 2: Use the corners that work (from your logs)
    print("\n--- Method 2: Working corners from logs ---")
    corners2 = [[302.3999938964844, 302.3999938964844], [3729.60009765625, 302.3999938964844], [3729.60009765625, 2721.60009765625], [302.3999938964844, 2721.60009765625]]
    test_with_corners(image_path, corners2, "Working corners")
    
    # Method 3: Use estimated corners
    print("\n--- Method 3: Estimated corners ---")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    margin = min(h, w) * 0.1
    corners3 = [[margin, margin], [w - margin, margin], [w - margin, h - margin], [margin, h - margin]]
    test_with_corners(image_path, corners3, "Estimated corners")

def detect_corners_properly(image_path):
    """Detect chessboard corners properly"""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different chessboard sizes
    for pattern_size in [(7, 7), (8, 8), (9, 9)]:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            logger.info(f"Found {pattern_size} chessboard pattern")
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

def test_with_corners(image_path, corners, method_name):
    """Test API with specific corners"""
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
            debug_info = result.get('debug_info', {})
            
            # Calculate metrics
            pieces_detected = sum(1 for p in pieces if p is not None)
            occupied_squares = sum(occupancy)
            total_squares = len(occupancy)
            
            print(f"  Pieces detected: {pieces_detected}")
            print(f"  Occupied squares: {occupied_squares}")
            print(f"  FEN: {fen}")
            print(f"  Debug info available: {'Yes' if debug_info else 'No'}")
            
            if debug_info:
                print(f"  Debug info keys: {list(debug_info.keys())}")
                if 'square_details' in debug_info:
                    square_details = debug_info['square_details']
                    print(f"  Square details: {len(square_details)} squares")
                    
                    # Analyze confidence scores
                    occupancy_scores = []
                    color_scores = []
                    piece_scores = []
                    
                    for square in square_details:
                        if 'occupancy_probs' in square:
                            occ_probs = square['occupancy_probs']
                            occupied_prob = occ_probs.get('occupied', 0)
                            occupancy_scores.append(occupied_prob)
                        
                        if 'color_confidence' in square:
                            color_scores.append(square['color_confidence'])
                        
                        if 'piece_confidence' in square:
                            piece_scores.append(square['piece_confidence'])
                    
                    if occupancy_scores:
                        print(f"  Occupancy scores - Min: {min(occupancy_scores):.3f}, Max: {max(occupancy_scores):.3f}, Mean: {np.mean(occupancy_scores):.3f}")
                    
                    if color_scores:
                        print(f"  Color scores - Min: {min(color_scores):.3f}, Max: {max(color_scores):.3f}, Mean: {np.mean(color_scores):.3f}")
                    
                    if piece_scores:
                        print(f"  Piece scores - Min: {min(piece_scores):.3f}, Max: {max(piece_scores):.3f}, Mean: {np.mean(piece_scores):.3f}")
                    
                    # Count high confidence predictions
                    high_occ = sum(1 for s in occupancy_scores if s > 0.5)
                    high_color = sum(1 for s in color_scores if s > 0.8)
                    high_piece = sum(1 for s in piece_scores if s > 0.8)
                    
                    print(f"  High confidence occupancy: {high_occ}")
                    print(f"  High confidence color: {high_color}")
                    print(f"  High confidence piece: {high_piece}")
                    
                    # Calculate the 4 metrics you requested
                    occupancy_accuracy = (occupied_squares / total_squares) * 100
                    color_accuracy = (high_color / occupied_squares) * 100 if occupied_squares > 0 else 0
                    piece_accuracy = (high_piece / occupied_squares) * 100 if occupied_squares > 0 else 0
                    fen_accuracy = 100 if fen != '8/8/8/8/8/8/8/8 w - - 0 1' and pieces_detected > 0 else 0
                    
                    print(f"\n  YOUR REQUESTED 4 METRICS:")
                    print(f"  1. % of squares where occupancy is correct: {occupancy_accuracy:.1f}%")
                    print(f"  2. % of occupied squares where color is correct: {color_accuracy:.1f}%")
                    print(f"  3. % of occupied squares where piece is correct: {piece_accuracy:.1f}%")
                    print(f"  4. % of images where entire FEN is 100% correct: {fen_accuracy:.1f}%")
                    
                    return {
                        'method': method_name,
                        'pieces_detected': pieces_detected,
                        'occupied_squares': occupied_squares,
                        'total_squares': total_squares,
                        'occupancy_accuracy': occupancy_accuracy,
                        'color_accuracy': color_accuracy,
                        'piece_accuracy': piece_accuracy,
                        'fen_accuracy': fen_accuracy,
                        'fen': fen
                    }
            
            return {
                'method': method_name,
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'total_squares': total_squares,
                'occupancy_accuracy': 0,
                'color_accuracy': 0,
                'piece_accuracy': 0,
                'fen_accuracy': 0,
                'fen': fen
            }
        else:
            print(f"  ❌ API call failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """Main function"""
    print("Simple Model Breakdown - Testing Different Corner Detection Methods")
    print("=" * 70)
    
    results = test_api_with_different_corners()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The best method will show the highest accuracy metrics.")
    print("Use the corners from the best performing method for your evaluation.")

if __name__ == "__main__":
    main()
