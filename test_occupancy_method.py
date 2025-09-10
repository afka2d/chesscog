#!/usr/bin/env python3
"""
Test script to debug occupancy detection method
"""

import numpy as np
import cv2
import json
import chess
from chesscog.recognition.recognition import ChessRecognizer

def test_occupancy_method():
    print("🧪 TESTING OCCUPANCY DETECTION METHOD")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    corners_list = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    corners_array = np.array(corners_list, dtype=np.float32)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return
    
    print(f"✅ Loaded test image: {img_path}")
    print(f"   Image shape: {img.shape}")
    print(f"✅ Loaded corners: {corners_list}")
    
    # Initialize recognizer
    recognizer = ChessRecognizer()
    print("✅ ChessRecognizer initialized")
    
    # Test the _classify_occupancy method directly
    print("\n🔍 Testing _classify_occupancy method directly...")
    try:
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners_array)
        print(f"✅ _classify_occupancy completed")
        print(f"   Occupancy shape: {occupancy.shape}")
        print(f"   Occupancy type: {type(occupancy)}")
        print(f"   Detected {np.sum(occupancy)} occupied squares out of 64")
        
        if occupancy.ndim == 1:
            occupancy_2d = occupancy.reshape(8, 8)
            print("   Converted to 2D for visualization:")
            for r in range(8):
                print("      " + "".join(["X" if occupancy_2d[r, c] else "." for c in range(8)]))
        else:
            print("   Already 2D, showing pattern:")
            for r in range(8):
                print("      " + "".join(["X" if occupancy[r, c] else "." for c in range(8)]))
                
    except Exception as e:
        print(f"❌ Error in _classify_occupancy: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_occupancy_method()
