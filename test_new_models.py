#!/usr/bin/env python3
"""
Test the newly trained models on a sample from our dataset
"""

import cv2
import numpy as np
from pathlib import Path
import json
import chess
from chesscog.recognition import ChessRecognizer

def test_model_on_sample():
    """Test the new models on a sample image from our dataset."""
    
    # Load a sample image and annotation
    sample_image = "grey_background_dataset/images/test/IMG_4752.JPG"
    sample_annotation = "grey_background_dataset/annotations/test/IMG_4752.json"
    
    if not Path(sample_image).exists():
        print(f"Sample image not found: {sample_image}")
        return
    
    if not Path(sample_annotation).exists():
        print(f"Sample annotation not found: {sample_annotation}")
        return
    
    # Load ground truth
    with open(sample_annotation, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Testing model on: {sample_image}")
    print(f"Ground truth FEN: {ground_truth['fen']}")
    
    # Load image
    img = cv2.imread(sample_image)
    if img is None:
        print(f"Could not load image: {sample_image}")
        return
    
    # Initialize recognizer with new models
    recognizer = ChessRecognizer()
    
    try:
        # Use the manual corners from annotation for accurate comparison
        corners = ground_truth['corners']
        print(f"Using manual corners: {corners}")
        
        # Predict the board state
        predicted_fen = recognizer.predict(img, corners)
        
        print(f"Predicted FEN:    {predicted_fen}")
        print(f"Ground truth FEN: {ground_truth['fen']}")
        
        # Compare FENs (just the board position part)
        if predicted_fen and ground_truth['fen']:
            pred_board = predicted_fen.split()[0]
            true_board = ground_truth['fen'].split()[0]
            
            if pred_board == true_board:
                print("✅ PERFECT MATCH!")
            else:
                print("❌ Mismatch in board position")
                
                # Count piece-level accuracy
                pred_pieces = []
                true_pieces = []
                
                # Parse board positions
                pred_board_obj = chess.Board(predicted_fen)
                true_board_obj = chess.Board(ground_truth['fen'])
                
                correct_squares = 0
                total_squares = 64
                
                for square in chess.SQUARES:
                    pred_piece = pred_board_obj.piece_at(square)
                    true_piece = true_board_obj.piece_at(square)
                    
                    if pred_piece == true_piece:
                        correct_squares += 1
                
                accuracy = correct_squares / total_squares * 100
                print(f"Square-level accuracy: {accuracy:.1f}% ({correct_squares}/{total_squares})")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_samples():
    """Test on multiple samples to get overall performance."""
    test_dir = Path("grey_background_dataset/images/test")
    annotations_dir = Path("grey_background_dataset/annotations/test")
    
    if not test_dir.exists():
        print("Test directory not found")
        return
    
    recognizer = ChessRecognizer()
    
    total_tested = 0
    perfect_matches = 0
    total_accuracy = 0
    
    for img_file in test_dir.glob("*.JPG"):
        annotation_file = annotations_dir / (img_file.stem + ".json")
        
        if not annotation_file.exists():
            continue
            
        try:
            # Load annotation
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            if not annotation.get('fen') or annotation['fen'] == "8/8/8/8/8/8/8/8 w - - 0 1":
                continue
                
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Predict
            corners = annotation['corners']
            predicted_fen = recognizer.predict(img, corners)
            
            if predicted_fen:
                pred_board = predicted_fen.split()[0]
                true_board = annotation['fen'].split()[0]
                
                if pred_board == true_board:
                    perfect_matches += 1
                
                # Calculate square-level accuracy
                pred_board_obj = chess.Board(predicted_fen)
                true_board_obj = chess.Board(annotation['fen'])
                
                correct_squares = sum(1 for square in chess.SQUARES 
                                    if pred_board_obj.piece_at(square) == true_board_obj.piece_at(square))
                
                accuracy = correct_squares / 64 * 100
                total_accuracy += accuracy
                
                total_tested += 1
                
                if total_tested <= 5:  # Show first 5 results
                    print(f"\n{img_file.name}:")
                    print(f"  Predicted: {predicted_fen}")
                    print(f"  Actual:    {annotation['fen']}")
                    print(f"  Accuracy:  {accuracy:.1f}%")
                    print(f"  Perfect:   {'✅' if pred_board == true_board else '❌'}")
                
        except Exception as e:
            print(f"Error testing {img_file.name}: {e}")
            continue
    
    if total_tested > 0:
        avg_accuracy = total_accuracy / total_tested
        perfect_rate = perfect_matches / total_tested * 100
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"Images tested: {total_tested}")
        print(f"Perfect matches: {perfect_matches} ({perfect_rate:.1f}%)")
        print(f"Average square accuracy: {avg_accuracy:.1f}%")
    else:
        print("No valid test images found")

if __name__ == "__main__":
    print("Testing newly trained models...")
    print("=" * 50)
    
    # Test single sample first
    test_model_on_sample()
    
    print("\n" + "=" * 50)
    print("Testing multiple samples...")
    
    # Test multiple samples
    test_multiple_samples()