import cv2
import numpy as np
from pathlib import Path
from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN
import chess

def test_position_extractor_simple():
    """Test the chess position extractor on the provided image using the simple predict method."""
    
    # Load configuration
    print("Loading corner detection configuration...")
    cfg = CN.load_yaml_with_base("config/corner_detection.yaml")
    
    # Load recognizer
    print("Loading chess recognizer...")
    recognizer = ChessRecognizer(Path("models"))
    
    # Load the chess board image
    image_path = "IMG_4540.jpeg"  # Using an existing test image
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert BGR to RGB (OpenCV loads as BGR, but chesscog expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Image loaded successfully")
    print(f"Image shape: {img_rgb.shape}")
    
    try:
        # Test corner detection
        print("\n=== CORNER DETECTION ===")
        corners, debug_images = find_corners(cfg, img_rgb)
        print(f"Corners detected: {corners}")
        
        # Test full recognition (simple version without debug images)
        print("\n=== FULL RECOGNITION ===")
        board, corners_final = recognizer.predict(img_rgb, chess.WHITE)
        
        # Get results
        fen = board.fen()
        ascii_board = str(board)
        lichess_url = f"https://lichess.org/editor/{fen}?color=white"
        legal = board.is_valid()
        
        print("\n=== RECOGNITION RESULTS ===")
        print(f"FEN: {fen}")
        print(f"ASCII Board:\n{ascii_board}")
        print(f"Lichess URL: {lichess_url}")
        print(f"Legal Position: {legal}")
        print(f"Final Corners: {corners_final}")
        
        # Save corner detection debug images
        print("\n=== SAVING CORNER DETECTION DEBUG IMAGES ===")
        debug_output_dir = "debug_outputs"
        Path(debug_output_dir).mkdir(exist_ok=True)
        
        for key, img in debug_images.items():
            if isinstance(img, np.ndarray):
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                filename = f"position_test_corners_{key}.png"
                filepath = Path(debug_output_dir) / filename
                cv2.imwrite(str(filepath), img_bgr)
                print(f"Saved debug image: {filepath}")
        
        # Analyze the detected position
        print("\n=== POSITION ANALYSIS ===")
        piece_count = len([piece for piece in board.piece_map().values()])
        print(f"Total pieces detected: {piece_count}")
        
        # Show piece positions
        piece_map = board.piece_map()
        if piece_map:
            print("Detected pieces:")
            for square, piece in piece_map.items():
                square_name = chess.square_name(square)
                piece_name = piece.symbol()
                color = "White" if piece.color else "Black"
                print(f"  {square_name}: {color} {piece_name}")
        else:
            print("No pieces detected")
        
        # Test with black perspective as well
        print("\n=== TESTING BLACK PERSPECTIVE ===")
        board_black, corners_black = recognizer.predict(img_rgb, chess.BLACK)
        fen_black = board_black.fen()
        ascii_black = str(board_black)
        legal_black = board_black.is_valid()
        
        print(f"Black perspective FEN: {fen_black}")
        print(f"Black perspective ASCII:\n{ascii_black}")
        print(f"Black perspective Legal: {legal_black}")
        
        return board, corners_final, debug_images
        
    except Exception as e:
        print(f"Error during recognition: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    test_position_extractor_simple() 