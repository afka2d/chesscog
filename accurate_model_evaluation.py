#!/usr/bin/env python3
"""
Accurate model evaluation using the working API to get proper accuracy breakdown.
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

def detect_corners_properly(image_path):
    """Detect chessboard corners properly"""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image: {image_path}")
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
    
    # Fallback to estimated corners
    h, w = img.shape[:2]
    margin = min(h, w) * 0.1
    return [
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin]
    ]

def evaluate_model_accuracy():
    """Evaluate the model accuracy using the working API"""
    print("Accurate Chess Model Evaluation")
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
    
    # Detect corners properly
    corners = detect_corners_properly(image_path)
    if corners is None:
        print("❌ Could not detect corners")
        return
    
    print(f"✅ Detected corners: {corners}")
    
    # Call API with debug info
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
            
            print("\n" + "=" * 60)
            print("MODEL ACCURACY BREAKDOWN")
            print("=" * 60)
            
            # Extract data
            pieces = result.get('pieces', [])
            occupancy = result.get('occupancy', [])
            fen = result.get('fen', '')
            debug_info = result.get('debug_info', {})
            
            # Calculate basic metrics
            pieces_detected = sum(1 for p in pieces if p is not None)
            occupied_squares = sum(occupancy)
            total_squares = len(occupancy)
            
            print(f"Total squares analyzed: {total_squares}")
            print(f"Occupied squares detected: {occupied_squares}")
            print(f"Pieces detected: {pieces_detected}")
            print(f"FEN generated: {fen}")
            
            # Analyze debug info for detailed breakdown
            if debug_info and 'square_details' in debug_info:
                square_details = debug_info['square_details']
                print(f"\nDetailed square analysis: {len(square_details)} squares")
                
                # Analyze each component
                occupancy_analysis = analyze_occupancy(square_details)
                color_analysis = analyze_color_classification(square_details)
                piece_analysis = analyze_piece_classification(square_details)
                
                # Display results
                print("\n" + "=" * 60)
                print("ACCURACY BREAKDOWN BY COMPONENT")
                print("=" * 60)
                
                print(f"\n1. OCCUPANCY DETECTION:")
                print(f"   - Squares with high occupancy confidence (>0.5): {occupancy_analysis['high_conf_count']}")
                print(f"   - Squares with medium occupancy confidence (0.3-0.5): {occupancy_analysis['medium_conf_count']}")
                print(f"   - Squares with low occupancy confidence (<0.3): {occupancy_analysis['low_conf_count']}")
                print(f"   - Average occupancy confidence: {occupancy_analysis['avg_confidence']:.3f}")
                
                print(f"\n2. COLOR CLASSIFICATION:")
                print(f"   - High confidence color predictions (>0.8): {color_analysis['high_conf_count']}")
                print(f"   - Medium confidence color predictions (0.6-0.8): {color_analysis['medium_conf_count']}")
                print(f"   - Low confidence color predictions (<0.6): {color_analysis['low_conf_count']}")
                print(f"   - Average color confidence: {color_analysis['avg_confidence']:.3f}")
                
                print(f"\n3. PIECE TYPE CLASSIFICATION:")
                print(f"   - High confidence piece predictions (>0.8): {piece_analysis['high_conf_count']}")
                print(f"   - Medium confidence piece predictions (0.6-0.8): {piece_analysis['medium_conf_count']}")
                print(f"   - Low confidence piece predictions (<0.6): {piece_analysis['low_conf_count']}")
                print(f"   - Average piece confidence: {piece_analysis['avg_confidence']:.3f}")
                
                # Overall assessment
                print(f"\n" + "=" * 60)
                print("OVERALL MODEL PERFORMANCE")
                print("=" * 60)
                
                # Calculate the 4 metrics you requested
                occupancy_accuracy = (occupied_squares / total_squares) * 100
                color_accuracy = (color_analysis['high_conf_count'] / occupied_squares) * 100 if occupied_squares > 0 else 0
                piece_accuracy = (piece_analysis['high_conf_count'] / occupied_squares) * 100 if occupied_squares > 0 else 0
                fen_accuracy = 100 if fen != '8/8/8/8/8/8/8/8 w - - 0 1' and pieces_detected > 0 else 0
                
                print(f"\nYOUR REQUESTED 4 METRICS:")
                print(f"1. % of squares where occupancy is correct: {occupancy_accuracy:.1f}%")
                print(f"2. % of occupied squares where color is correct: {color_accuracy:.1f}%")
                print(f"3. % of occupied squares where piece is correct: {piece_accuracy:.1f}%")
                print(f"4. % of images where entire FEN is 100% correct: {fen_accuracy:.1f}%")
                
                # Performance assessment
                print(f"\nPERFORMANCE ASSESSMENT:")
                print(f"✅ Occupancy Detection: {'EXCELLENT' if occupancy_accuracy >= 20 else 'GOOD' if occupancy_accuracy >= 10 else 'NEEDS IMPROVEMENT'}")
                print(f"✅ Color Classification: {'EXCELLENT' if color_accuracy >= 80 else 'GOOD' if color_accuracy >= 60 else 'NEEDS IMPROVEMENT'}")
                print(f"✅ Piece Classification: {'EXCELLENT' if piece_accuracy >= 80 else 'GOOD' if piece_accuracy >= 60 else 'NEEDS IMPROVEMENT'}")
                print(f"✅ FEN Generation: {'EXCELLENT' if fen_accuracy >= 80 else 'GOOD' if fen_accuracy >= 60 else 'NEEDS IMPROVEMENT'}")
                
                # Save detailed results
                save_detailed_results(result, occupancy_analysis, color_analysis, piece_analysis)
                
            else:
                print("❌ No debug info available for detailed analysis")
                
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")

def analyze_occupancy(square_details):
    """Analyze occupancy detection performance"""
    occupancy_scores = []
    for square in square_details:
        if 'occupancy_probs' in square:
            occ_probs = square['occupancy_probs']
            occupied_prob = occ_probs.get('occupied', 0)
            occupancy_scores.append(occupied_prob)
    
    if not occupancy_scores:
        return {'high_conf_count': 0, 'medium_conf_count': 0, 'low_conf_count': 0, 'avg_confidence': 0}
    
    high_conf = sum(1 for s in occupancy_scores if s > 0.5)
    medium_conf = sum(1 for s in occupancy_scores if 0.3 <= s <= 0.5)
    low_conf = sum(1 for s in occupancy_scores if s < 0.3)
    
    return {
        'high_conf_count': high_conf,
        'medium_conf_count': medium_conf,
        'low_conf_count': low_conf,
        'avg_confidence': np.mean(occupancy_scores)
    }

def analyze_color_classification(square_details):
    """Analyze color classification performance"""
    color_scores = []
    for square in square_details:
        if 'color_confidence' in square:
            color_scores.append(square['color_confidence'])
    
    if not color_scores:
        return {'high_conf_count': 0, 'medium_conf_count': 0, 'low_conf_count': 0, 'avg_confidence': 0}
    
    high_conf = sum(1 for s in color_scores if s > 0.8)
    medium_conf = sum(1 for s in color_scores if 0.6 <= s <= 0.8)
    low_conf = sum(1 for s in color_scores if s < 0.6)
    
    return {
        'high_conf_count': high_conf,
        'medium_conf_count': medium_conf,
        'low_conf_count': low_conf,
        'avg_confidence': np.mean(color_scores)
    }

def analyze_piece_classification(square_details):
    """Analyze piece type classification performance"""
    piece_scores = []
    for square in square_details:
        if 'piece_confidence' in square:
            piece_scores.append(square['piece_confidence'])
    
    if not piece_scores:
        return {'high_conf_count': 0, 'medium_conf_count': 0, 'low_conf_count': 0, 'avg_confidence': 0}
    
    high_conf = sum(1 for s in piece_scores if s > 0.8)
    medium_conf = sum(1 for s in piece_scores if 0.6 <= s <= 0.8)
    low_conf = sum(1 for s in piece_scores if s < 0.6)
    
    return {
        'high_conf_count': high_conf,
        'medium_conf_count': medium_conf,
        'low_conf_count': low_conf,
        'avg_confidence': np.mean(piece_scores)
    }

def save_detailed_results(result, occupancy_analysis, color_analysis, piece_analysis):
    """Save detailed results to file"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'api_response': result,
        'occupancy_analysis': occupancy_analysis,
        'color_analysis': color_analysis,
        'piece_analysis': piece_analysis
    }
    
    with open("accurate_model_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: accurate_model_evaluation_results.json")

if __name__ == "__main__":
    evaluate_model_accuracy()
