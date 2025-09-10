#!/usr/bin/env python3
"""
Debug the piece extraction process to see why pieces are being misclassified.
"""

import numpy as np
from PIL import Image
import chess
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path
import cv2

def debug_piece_extraction():
    """Debug the piece extraction process."""
    print("🔍 Debugging Piece Extraction Process")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Test corners
    corners = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    print(f"📸 Image loaded: {img_array.shape}")
    print(f"📐 Corners: {corners.shape}")
    
    # Initialize classifiers
    print("\n🔧 Initializing classifiers...")
    piece_classifier = SimplePieceClassifier(Path("models"))
    recognizer = ChessRecognizer(Path("models"))
    
    # Get real occupancy from ChessRecognizer
    print("\n🎯 Getting real occupancy...")
    board, detected_corners = recognizer.predict(img_array, chess.WHITE)
    
    # Convert board to occupancy array
    occupancy = np.zeros(64, dtype=bool)
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            occupancy[square] = True
    
    occupied_count = np.sum(occupancy)
    print(f"   Occupied squares: {occupied_count}/64")
    
    # Debug the piece extraction process
    print(f"\n🔍 DEBUGGING PIECE EXTRACTION:")
    
    # Get the perspective transformation
    from chesscog.core.board_detection import find_corners
    from chesscog.core.board_detection import get_chessboard_corners
    
    try:
        # Find corners using the same method as the classifier
        corners_detected = get_chessboard_corners(img_array)
        print(f"   Detected corners: {corners_detected.shape if corners_detected is not None else 'None'}")
        
        if corners_detected is not None:
            # Use detected corners
            corners_to_use = corners_detected
        else:
            # Use provided corners
            corners_to_use = corners
            print(f"   Using provided corners: {corners_to_use}")
        
        # Extract pieces from occupied squares
        piece_imgs = []
        occupied_squares = []
        
        for i, is_occupied in enumerate(occupancy):
            if is_occupied:
                rank, file = i // 8, i % 8
                occupied_squares.append((rank, file))
                
                # Extract piece image using the same method as SimplePieceClassifier
                try:
                    # This is a simplified version - the actual extraction is more complex
                    piece_img = img  # This is where the issue might be
                    piece_imgs.append(piece_img)
                    print(f"   Square {i} (rank {rank}, file {file}): Extracted piece image {piece_img.size}")
                except Exception as e:
                    print(f"   Square {i}: Error extracting piece - {e}")
        
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"   Occupied squares: {len(occupied_squares)}")
        print(f"   Piece images extracted: {len(piece_imgs)}")
        print(f"   Occupied squares: {occupied_squares}")
        
        # Test classification on extracted pieces
        if piece_imgs:
            print(f"\n🎲 TESTING CLASSIFICATION ON EXTRACTED PIECES:")
            try:
                # Apply transforms
                piece_imgs_transformed = [piece_classifier._pieces_transforms(img) for img in piece_imgs]
                piece_imgs_tensor = torch.stack(piece_imgs_transformed)
                
                # Get predictions
                with torch.no_grad():
                    predictions = piece_classifier._pieces_model(piece_imgs_tensor)
                    predicted_classes = predictions.argmax(axis=-1).cpu().numpy()
                    piece_names = piece_classifier._piece_classes[predicted_classes]
                
                # Analyze results
                piece_names_str = []
                for i, piece in enumerate(piece_names):
                    if hasattr(piece, 'symbol'):
                        piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                        piece_names_str.append(piece_name)
                        square = occupied_squares[i]
                        print(f"   Square {square}: {piece_name}")
                    else:
                        piece_names_str.append(str(piece))
                
                # Check for bias
                from collections import Counter
                piece_counts = Counter(piece_names_str)
                print(f"\n📈 CLASSIFICATION RESULTS:")
                print(f"   Piece counts: {dict(piece_counts)}")
                
                pawn_count = sum(1 for name in piece_names_str if 'p' in name.lower())
                pawn_ratio = pawn_count / len(piece_names_str) if piece_names_str else 0
                print(f"   Pawn ratio: {pawn_count}/{len(piece_names_str)} ({pawn_ratio*100:.1f}%)")
                
            except Exception as e:
                print(f"   Error in classification: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"   Error in piece extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    debug_piece_extraction()