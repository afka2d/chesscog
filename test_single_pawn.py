import cv2
import numpy as np
from pathlib import Path
from chesscog.recognition.recognition import ChessRecognizer
import chess

def test_single_pawn_image(image_path):
    """Test the chess position extractor on an image with a single black pawn on e4."""
    
    print("Testing chess position extraction for image with single black pawn on e4")
    print("=" * 70)
    
    # Load recognizer
    print("Loading chess recognizer...")
    recognizer = ChessRecognizer(Path("models"))
    
    # Load the chess board image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    # Convert BGR to RGB (OpenCV loads as BGR, but chesscog expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Image loaded successfully")
    print(f"Image shape: {img_rgb.shape}")
    
    try:
        # Test recognition with white perspective
        print("\n=== WHITE PERSPECTIVE RECOGNITION ===")
        board_white, corners_white = recognizer.predict(img_rgb, chess.WHITE)
        
        fen_white = board_white.fen()
        ascii_white = str(board_white)
        legal_white = board_white.is_valid()
        
        print(f"FEN: {fen_white}")
        print(f"ASCII Board:\n{ascii_white}")
        print(f"Legal Position: {legal_white}")
        
        # Test recognition with black perspective
        print("\n=== BLACK PERSPECTIVE RECOGNITION ===")
        board_black, corners_black = recognizer.predict(img_rgb, chess.BLACK)
        
        fen_black = board_black.fen()
        ascii_black = str(board_black)
        legal_black = board_black.is_valid()
        
        print(f"FEN: {fen_black}")
        print(f"ASCII Board:\n{ascii_black}")
        print(f"Legal Position: {legal_black}")
        
        # Analyze results
        print("\n=== ANALYSIS ===")
        
        # Check for expected position (single black pawn on e4)
        expected_fen_white = "rnbqkbnr/pppppppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        expected_fen_black = "RNBQKBNR/PPPPPPPP/8/8/4P3/8/pppppppp/rnbqkbnr w KQkq - 0 1"
        
        print("Expected position (white perspective): Single black pawn on e4")
        print(f"Expected FEN (white): {expected_fen_white}")
        print(f"Detected FEN (white): {fen_white}")
        print(f"FEN matches expected: {fen_white == expected_fen_white}")
        
        print("\nExpected position (black perspective): Single white pawn on e5")
        print(f"Expected FEN (black): {expected_fen_black}")
        print(f"Detected FEN (black): {fen_black}")
        print(f"FEN matches expected: {fen_black == expected_fen_black}")
        
        # Count pieces
        piece_count_white = len([piece for piece in board_white.piece_map().values()])
        piece_count_black = len([piece for piece in board_black.piece_map().values()])
        
        print(f"\nPieces detected (white perspective): {piece_count_white}")
        print(f"Pieces detected (black perspective): {piece_count_black}")
        
        # Show piece positions for white perspective
        piece_map_white = board_white.piece_map()
        if piece_map_white:
            print("\nDetected pieces (white perspective):")
            for square, piece in piece_map_white.items():
                square_name = chess.square_name(square)
                piece_name = piece.symbol()
                color = "White" if piece.color else "Black"
                print(f"  {square_name}: {color} {piece_name}")
        
        # Show piece positions for black perspective
        piece_map_black = board_black.piece_map()
        if piece_map_black:
            print("\nDetected pieces (black perspective):")
            for square, piece in piece_map_black.items():
                square_name = chess.square_name(square)
                piece_name = piece.symbol()
                color = "White" if piece.color else "Black"
                print(f"  {square_name}: {color} {piece_name}")
        
        # Generate Lichess URLs
        lichess_url_white = f"https://lichess.org/editor/{fen_white}?color=white"
        lichess_url_black = f"https://lichess.org/editor/{fen_black}?color=black"
        
        print(f"\nLichess URL (white perspective): {lichess_url_white}")
        print(f"Lichess URL (black perspective): {lichess_url_black}")
        
        return board_white, board_black
        
    except Exception as e:
        print(f"Error during recognition: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to test single pawn image."""
    
    # You can change this to the path of your chess board image
    image_path = "IMG_4540.jpeg"  # Default test image
    
    # If a command line argument is provided, use it as the image path
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    result = test_single_pawn_image(image_path)
    
    if result[0] is not None:
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("TEST FAILED!")
        print("=" * 70)

if __name__ == "__main__":
    main() 