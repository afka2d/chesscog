#!/usr/bin/env python3
"""
Detailed evaluation script that analyzes confidence scores and thresholds.
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

def analyze_confidence_scores(debug_info):
    """Analyze confidence scores from debug info"""
    if not debug_info or 'square_details' not in debug_info:
        return None
    
    square_details = debug_info['square_details']
    
    # Extract confidence scores
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
    
    return {
        'occupancy_scores': occupancy_scores,
        'color_scores': color_scores,
        'piece_scores': piece_scores
    }

def evaluate_image_detailed(image_path):
    """Evaluate a single image with detailed analysis"""
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
            
            # Analyze confidence scores
            confidence_analysis = analyze_confidence_scores(debug_info)
            
            logger.info(f"  Pieces detected: {pieces_detected}")
            logger.info(f"  Occupied squares: {occupied_squares}")
            logger.info(f"  Processing time: {processing_time:.3f}s")
            logger.info(f"  FEN: {fen}")
            
            if confidence_analysis:
                occ_scores = confidence_analysis['occupancy_scores']
                color_scores = confidence_analysis['color_scores']
                piece_scores = confidence_analysis['piece_scores']
                
                if occ_scores:
                    logger.info(f"  Occupancy scores - Min: {min(occ_scores):.3f}, Max: {max(occ_scores):.3f}, Mean: {np.mean(occ_scores):.3f}")
                
                if color_scores:
                    logger.info(f"  Color scores - Min: {min(color_scores):.3f}, Max: {max(color_scores):.3f}, Mean: {np.mean(color_scores):.3f}")
                
                if piece_scores:
                    logger.info(f"  Piece scores - Min: {min(piece_scores):.3f}, Max: {max(piece_scores):.3f}, Mean: {np.mean(piece_scores):.3f}")
            
            return {
                'image': Path(image_path).name,
                'pieces_detected': pieces_detected,
                'occupied_squares': occupied_squares,
                'processing_time': processing_time,
                'fen': fen,
                'success': result.get('success', False),
                'confidence_analysis': confidence_analysis
            }
        else:
            logger.error(f"API call failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error evaluating {image_path}: {e}")
        return None

def main():
    """Main evaluation function"""
    print("Chess Model Detailed Evaluation")
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
    
    # Evaluate each image
    results = []
    
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)} ---")
        result = evaluate_image_detailed(str(image_path))
        if result:
            results.append(result)
        
        # Small delay
        time.sleep(1)
    
    # Summary
    if results:
        print("\n" + "=" * 40)
        print("DETAILED EVALUATION SUMMARY")
        print("=" * 40)
        
        total_pieces = sum(r['pieces_detected'] for r in results)
        total_occupied = sum(r['occupied_squares'] for r in results)
        avg_time = np.mean([r['processing_time'] for r in results])
        
        print(f"Images processed: {len(results)}")
        print(f"Total pieces detected: {total_pieces}")
        print(f"Total occupied squares: {total_occupied}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        # Confidence analysis
        all_occ_scores = []
        all_color_scores = []
        all_piece_scores = []
        
        for result in results:
            if result['confidence_analysis']:
                all_occ_scores.extend(result['confidence_analysis']['occupancy_scores'])
                all_color_scores.extend(result['confidence_analysis']['color_scores'])
                all_piece_scores.extend(result['confidence_analysis']['piece_scores'])
        
        if all_occ_scores:
            print(f"\nOverall Occupancy Confidence:")
            print(f"  Min: {min(all_occ_scores):.3f}")
            print(f"  Max: {max(all_occ_scores):.3f}")
            print(f"  Mean: {np.mean(all_occ_scores):.3f}")
            print(f"  Std: {np.std(all_occ_scores):.3f}")
        
        if all_color_scores:
            print(f"\nOverall Color Confidence:")
            print(f"  Min: {min(all_color_scores):.3f}")
            print(f"  Max: {max(all_color_scores):.3f}")
            print(f"  Mean: {np.mean(all_color_scores):.3f}")
            print(f"  Std: {np.std(all_color_scores):.3f}")
        
        if all_piece_scores:
            print(f"\nOverall Piece Confidence:")
            print(f"  Min: {min(all_piece_scores):.3f}")
            print(f"  Max: {max(all_piece_scores):.3f}")
            print(f"  Mean: {np.mean(all_piece_scores):.3f}")
            print(f"  Std: {np.std(all_piece_scores):.3f}")
        
        # Save results
        with open("detailed_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: detailed_evaluation_results.json")
    else:
        print("No results to summarize")

if __name__ == "__main__":
    main()
