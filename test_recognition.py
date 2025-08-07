import cv2
import numpy as np
from pathlib import Path
from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN

def test_recognition():
    # Load configuration
    cfg = CN.load_yaml_with_base("config/corner_detection.yaml")
    
    # Load recognizer
    recognizer = ChessRecognizer(Path("models"))
    
    # Load image
    img = cv2.imread("debug_cropped_board.jpeg")
    if img is None:
        print("Failed to load image")
        return
    
    print("Image loaded successfully")
    print(f"Image shape: {img.shape}")
    
    try:
        # Test corner detection
        print("Testing corner detection...")
        corners, debug_images = find_corners(cfg, img)
        print(f"Corners detected: {corners}")
        
        # Test recognition
        print("Testing recognition...")
        board = recognizer.predict(img, "white")[0]
        
        fen = board.fen()
        ascii_board = str(board)
        lichess_url = f"https://lichess.org/editor/{fen}?color=white"
        legal = board.is_valid()
        
        print("\n=== RECOGNITION RESULTS ===")
        print(f"FEN: {fen}")
        print(f"ASCII Board:\n{ascii_board}")
        print(f"Lichess URL: {lichess_url}")
        print(f"Legal Position: {legal}")
        
    except Exception as e:
        print(f"Error during recognition: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recognition() 