#!/usr/bin/env python3
"""
Test script to debug what the API is actually calling
"""

import numpy as np
import cv2
import json
import chess
from chesscog.recognition.recognition import ChessRecognizer

def test_api_debug():
    print("🧪 TESTING API DEBUG")
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
    
    # Test what the API is actually calling
    print("\n🔍 Testing what the API calls...")
    
    # Simulate the exact API call
    try:
        print("Calling recognizer._classify_occupancy(img, chess.WHITE, corners_array)...")
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners_array)
        print(f"✅ _classify_occupancy completed")
        print(f"   Detected {np.sum(occupancy)} occupied squares out of 64")
        
    except Exception as e:
        print(f"❌ Error in _classify_occupancy: {e}")
        import traceback
        traceback.print_exc()
    
    # Test if there's a different method being called
    print("\n🔍 Testing if predict method is being called...")
    try:
        print("Calling recognizer.predict(img, chess.WHITE)...")
        board, corners = recognizer.predict(img, chess.WHITE)
        print(f"✅ predict completed")
        print(f"   Detected corners: {corners}")
        
    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_api_debug()
