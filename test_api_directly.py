#!/usr/bin/env python3
"""
Test the API directly to get accurate performance metrics.
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

def test_api_directly():
    """Test the API directly with a real image"""
    print("Testing API directly with IMG_4698.JPG")
    print("=" * 50)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            health_data = response.json()
            print(f"Health data: {health_data}")
        else:
            print("❌ API not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test with debug info
    try:
        response = requests.get("http://localhost:8001/debug/info", timeout=5)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"Debug info: {debug_data}")
        else:
            print("❌ Debug info not available")
    except Exception as e:
        print(f"❌ Error getting debug info: {e}")
    
    # Test image recognition
    image_path = "my_chess_images/train/images/IMG_4698.JPG"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\nTesting with image: {image_path}")
    
    # Detect corners
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image")
        return
    
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
        
        corners = [top_left, top_right, bottom_right, bottom_left]
        print(f"✅ Detected corners: {corners}")
    else:
        h, w = img.shape[:2]
        margin = min(h, w) * 0.1
        corners = [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]
        print(f"⚠️  Using estimated corners: {corners}")
    
    # Call API
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
            
            print("\n" + "=" * 50)
            print("API RESPONSE ANALYSIS")
            print("=" * 50)
            
            # Extract key data
            pieces = result.get('pieces', [])
            occupancy = result.get('occupancy', [])
            fen = result.get('fen', '')
            success = result.get('success', False)
            debug_info = result.get('debug_info', {})
            
            print(f"Success: {success}")
            print(f"FEN: {fen}")
            print(f"Pieces: {pieces}")
            print(f"Occupancy: {occupancy}")
            
            # Calculate metrics
            pieces_detected = sum(1 for p in pieces if p is not None)
            occupied_squares = sum(occupancy)
            total_squares = len(occupancy)
            
            print(f"\nMETRICS:")
            print(f"Total squares: {total_squares}")
            print(f"Occupied squares: {occupied_squares}")
            print(f"Pieces detected: {pieces_detected}")
            print(f"Occupancy rate: {(occupied_squares/total_squares)*100:.1f}%")
            print(f"Piece detection rate: {pieces_detected} pieces")
            
            # Analyze debug info
            if debug_info:
                print(f"\nDEBUG INFO:")
                print(f"Processing time: {debug_info.get('processing_time', 'N/A')}")
                print(f"Squares processed: {debug_info.get('squares_processed', 'N/A')}")
                print(f"Occupied squares: {debug_info.get('occupied_squares', 'N/A')}")
                print(f"Pieces detected: {debug_info.get('pieces_detected', 'N/A')}")
                
                # Analyze square details
                square_details = debug_info.get('square_details', [])
                if square_details:
                    print(f"\nSQUARE ANALYSIS:")
                    print(f"Total squares analyzed: {len(square_details)}")
                    
                    # Count confidence levels
                    high_conf_count = 0
                    medium_conf_count = 0
                    low_conf_count = 0
                    
                    for square in square_details:
                        if 'color_confidence' in square and 'piece_confidence' in square:
                            color_conf = square['color_confidence']
                            piece_conf = square['piece_confidence']
                            
                            if color_conf >= 0.8 and piece_conf >= 0.8:
                                high_conf_count += 1
                            elif color_conf >= 0.6 and piece_conf >= 0.6:
                                medium_conf_count += 1
                            else:
                                low_conf_count += 1
                    
                    print(f"High confidence squares: {high_conf_count}")
                    print(f"Medium confidence squares: {medium_conf_count}")
                    print(f"Low confidence squares: {low_conf_count}")
                    
                    # Show some examples
                    print(f"\nSAMPLE SQUARES:")
                    for i, square in enumerate(square_details[:5]):  # Show first 5
                        square_name = square.get('square', f'square_{i}')
                        occ_probs = square.get('occupancy_probs', {})
                        color_conf = square.get('color_confidence', 0)
                        piece_conf = square.get('piece_confidence', 0)
                        
                        print(f"  {square_name}: occ={occ_probs.get('occupied', 0):.3f}, color={color_conf:.3f}, piece={piece_conf:.3f}")
            
            # Overall assessment
            print(f"\nOVERALL ASSESSMENT:")
            print("-" * 30)
            
            if pieces_detected > 0:
                print("✅ Model is detecting pieces")
                if pieces_detected >= 5:
                    print("✅ Good piece detection rate")
                else:
                    print("⚠️  Low piece detection rate")
            else:
                print("❌ No pieces detected")
            
            if occupied_squares > 0:
                print("✅ Model is detecting occupied squares")
            else:
                print("❌ No occupied squares detected")
            
            if fen != '8/8/8/8/8/8/8/8 w - - 0 1':
                print("✅ Model is generating non-empty FEN")
            else:
                print("❌ Model is generating empty FEN")
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")

if __name__ == "__main__":
    test_api_directly()
