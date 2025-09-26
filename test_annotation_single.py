#!/usr/bin/env python3
"""
Test single image annotation
"""

import cv2
import json
import numpy as np
from pathlib import Path
import requests
import chess
from datetime import datetime

def test_single_image():
    """Test annotation on a single image"""
    
    # Test image path
    image_path = "/Users/tonyblum/Downloads/chess_set2_images/IMG_4573.JPG"
    
    print(f"🎯 Testing annotation on: {Path(image_path).name}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Test image loading
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        print(f"✅ Image loaded successfully: {image.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Test corner detection API
    try:
        print("🔍 Testing corner detection API...")
        with open(image_path, 'rb') as f:
            response = requests.post(
                "http://localhost:8005/detect_corners",
                files={'file': f},
                params={'time_budget': 2.0},
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                corners = data['corners']
                print(f"✅ Corners detected: {corners}")
                
                # Test FEN validation
                test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                try:
                    chess.Board(test_fen)
                    print(f"✅ FEN validation works: {test_fen}")
                except Exception as e:
                    print(f"❌ FEN validation failed: {e}")
                
                # Create annotation
                annotation = {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "chess_set": "set2",
                    "corners": corners,
                    "fen": test_fen,
                    "annotation_method": "test_single",
                    "corner_detection_api": "robust_port_8005",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save annotation
                output_dir = Path("./chess_set2_annotations")
                output_dir.mkdir(exist_ok=True)
                (output_dir / "annotations").mkdir(exist_ok=True)
                
                annotation_file = output_dir / "annotations" / f"{Path(image_path).stem}.json"
                with open(annotation_file, 'w') as f:
                    json.dump(annotation, f, indent=2)
                
                print(f"✅ Annotation saved: {annotation_file}")
                print("🎉 Single image annotation test successful!")
                
            else:
                print(f"❌ Corner detection failed: {data}")
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error calling API: {e}")

if __name__ == "__main__":
    test_single_image()
