#!/usr/bin/env python3
"""
Quick model improvement script with actionable recommendations.
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

def test_current_performance():
    """Test current model performance"""
    print("Testing Current Model Performance")
    print("=" * 40)
    
    # Find test images
    dataset_path = Path("my_chess_images/train/images")
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(dataset_path.glob(f"**/*{ext}"))
    
    if not images:
        print("❌ No images found to test")
        return None
    
    results = []
    
    for i, image_path in enumerate(images):
        print(f"\n--- Image {i+1}/{len(images)}: {Path(image_path).name} ---")
        
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
                
                pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
                occupied_squares = sum(result.get('occupancy', []))
                fen = result.get('fen', '')
                
                print(f"  Pieces detected: {pieces_detected}")
                print(f"  Occupied squares: {occupied_squares}")
                print(f"  FEN: {fen}")
                
                results.append({
                    'image': Path(image_path).name,
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'fen': fen
                })
            else:
                print(f"  ❌ API call failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return results

def analyze_results(results):
    """Analyze results and provide recommendations"""
    if not results:
        print("❌ No results to analyze")
        return
    
    print("\n" + "=" * 40)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 40)
    
    total_pieces = sum(r['pieces_detected'] for r in results)
    total_occupied = sum(r['occupied_squares'] for r in results)
    images_with_pieces = sum(1 for r in results if r['pieces_detected'] > 0)
    
    print(f"Images tested: {len(results)}")
    print(f"Images with pieces detected: {images_with_pieces}")
    print(f"Total pieces detected: {total_pieces}")
    print(f"Total occupied squares: {total_occupied}")
    
    if images_with_pieces == 0:
        print("\n❌ PROBLEM: No pieces detected in any image!")
        print("   This suggests the occupancy threshold is too high.")
        print("   RECOMMENDATION: Lower the occupancy threshold to 0.3 or 0.2")
        
    elif images_with_pieces < len(results):
        print(f"\n⚠️  PARTIAL SUCCESS: {images_with_pieces}/{len(results)} images detected pieces")
        print("   This suggests inconsistent performance.")
        print("   RECOMMENDATIONS:")
        print("   1. Lower the occupancy threshold")
        print("   2. Check image quality and lighting")
        print("   3. Verify corner detection accuracy")
        
    else:
        print(f"\n✅ SUCCESS: All {len(results)} images detected pieces!")
        print("   Your model is working well.")
        print("   RECOMMENDATIONS:")
        print("   1. Test with more diverse images")
        print("   2. Create ground truth annotations for accuracy measurement")
        print("   3. Consider optimizing processing time")
    
    # Specific recommendations based on results
    if total_pieces > 0:
        avg_pieces = total_pieces / len(results)
        print(f"\nAverage pieces per image: {avg_pieces:.1f}")
        
        if avg_pieces < 5:
            print("   ⚠️  Low piece count - may indicate threshold issues")
        elif avg_pieces > 20:
            print("   ⚠️  High piece count - may indicate false positives")
        else:
            print("   ✅ Reasonable piece count")

def main():
    """Main improvement function"""
    print("Chess Model Improvement Tool")
    print("=" * 40)
    
    # Check API
    if not check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Test current performance
    results = test_current_performance()
    
    # Analyze and provide recommendations
    analyze_results(results)
    
    # Next steps
    print("\n" + "=" * 40)
    print("NEXT STEPS")
    print("=" * 40)
    
    print("1. If no pieces detected:")
    print("   - Edit main_local_dev.py")
    print("   - Change line ~280: is_occupied = occupied_prob > 0.3")
    print("   - Restart API: ./start_local_dev.sh")
    print("   - Run this script again")
    
    print("\n2. If some pieces detected:")
    print("   - Test with more images")
    print("   - Create ground truth annotations")
    print("   - Fine-tune thresholds")
    
    print("\n3. If all pieces detected:")
    print("   - Test with more diverse images")
    print("   - Measure accuracy with ground truth")
    print("   - Optimize for production")

if __name__ == "__main__":
    main()
