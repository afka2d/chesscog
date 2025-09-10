#!/usr/bin/env python3
"""
Test the API directly without HTTP requests.
"""

import numpy as np
import chess
from PIL import Image
from main import CustomChessRecognizer
from pathlib import Path

def test_api_direct():
    """Test the API logic directly."""
    print("🧪 Testing API Logic Directly")
    print("=" * 40)
    
    # Load image
    img = Image.open('grey_background_dataset/images/test/IMG_4763.JPG').convert('RGB')
    img_array = np.array(img)
    
    # Test corners
    corners_array = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    # Initialize recognizer
    recognizer = CustomChessRecognizer(Path("models"))
    
    # Get occupancy using original recognizer
    print("1. Getting occupancy...")
    board, detected_corners = recognizer.predict(img_array, chess.WHITE)
    
    # Get occupancy from the board
    occupancy = np.zeros(64, dtype=bool)
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            occupancy[square] = True
    
    print(f"   Occupancy: {sum(occupancy)} occupied squares")
    
    # Convert occupancy to a simple Python list
    occupancy_list = occupancy.tolist()
    
    # Test the custom piece classifier
    print("2. Testing custom piece classifier...")
    try:
        pieces_2d = recognizer._classify_pieces(img_array, chess.WHITE, corners_array, occupancy_list)
        print(f"   Success! Got pieces_2d: {pieces_2d.shape}")
        
        # Convert pieces_2d to 1D list for API response
        pieces = []
        for rank in range(8):
            for file in range(8):
                piece = pieces_2d[rank, file]
                if piece is not None:
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    pieces.append(piece_name)
                else:
                    pieces.append(None)
        
        print(f"   Pieces: {len([p for p in pieces if p is not None])} detected")
        print(f"   Piece types: {set(p for p in pieces if p is not None)}")
        
        return True
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_direct()
    if success:
        print("\n🎉 API logic works correctly!")
    else:
        print("\n❌ API logic has issues!")
