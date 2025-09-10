#!/usr/bin/env python3
"""
Debug which model is being loaded and why it's biased towards pawns.
"""

import numpy as np
from PIL import Image
import chess
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def debug_model_loading():
    """Debug which model is being loaded."""
    print("🔍 Debugging Model Loading")
    print("=" * 40)
    
    # Initialize classifier
    print("🔧 Initializing piece classifier...")
    piece_classifier = SimplePieceClassifier(Path("models"))
    
    # Check what model was loaded
    print(f"\n📊 MODEL INFO:")
    print(f"   Model loaded: {hasattr(piece_classifier, '_pieces_model')}")
    print(f"   Transforms loaded: {hasattr(piece_classifier, '_pieces_transforms')}")
    print(f"   Piece classes: {getattr(piece_classifier, '_piece_classes', 'Not loaded')}")
    
    if hasattr(piece_classifier, '_piece_classes'):
        print(f"   Piece classes shape: {piece_classifier._piece_classes.shape}")
        print(f"   Piece classes: {piece_classifier._piece_classes}")
        
        # Check if the model is biased towards a specific class
        print(f"\n🎯 TESTING MODEL BIAS:")
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (64, 64), color='white')
        
        # Test with a simple image multiple times
        test_results = []
        for i in range(10):
            try:
                # Create dummy piece images
                piece_imgs = [dummy_img] * 5  # 5 pieces
                
                # Apply transforms
                piece_imgs = [piece_classifier._pieces_transforms(img) for img in piece_imgs]
                piece_imgs = torch.stack(piece_imgs)
                
                # Get predictions
                with torch.no_grad():
                    predictions = piece_classifier._pieces_model(piece_imgs)
                    predicted_classes = predictions.argmax(axis=-1).cpu().numpy()
                    piece_names = piece_classifier._piece_classes[predicted_classes]
                
                # Convert to strings
                piece_names_str = []
                for piece in piece_names:
                    if hasattr(piece, 'symbol'):
                        piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                        piece_names_str.append(piece_name)
                    else:
                        piece_names_str.append(str(piece))
                
                test_results.extend(piece_names_str)
                
            except Exception as e:
                print(f"   Error in test {i}: {e}")
                break
        
        if test_results:
            from collections import Counter
            piece_counts = Counter(test_results)
            print(f"   Test results (50 predictions): {dict(piece_counts)}")
            
            # Check for bias
            total_predictions = len(test_results)
            pawn_predictions = sum(1 for name in test_results if 'p' in name.lower())
            pawn_ratio = pawn_predictions / total_predictions if total_predictions > 0 else 0
            
            print(f"   Pawn predictions: {pawn_predictions}/{total_predictions} ({pawn_ratio*100:.1f}%)")
            
            if pawn_ratio > 0.7:
                print("   ⚠️  STRONG PAWN BIAS DETECTED!")
            elif pawn_ratio > 0.5:
                print("   ⚠️  MODERATE PAWN BIAS DETECTED!")
            else:
                print("   ✅ No significant pawn bias")
        
        # Check model confidence
        print(f"\n🎲 CHECKING MODEL CONFIDENCE:")
        try:
            # Create a single dummy image
            dummy_img = Image.new('RGB', (64, 64), color='white')
            piece_img = piece_classifier._pieces_transforms(dummy_img)
            piece_img = torch.stack([piece_img])
            
            with torch.no_grad():
                predictions = piece_classifier._pieces_model(piece_img)
                probabilities = torch.softmax(predictions, dim=1)
                max_prob = probabilities.max().item()
                avg_prob = probabilities.mean().item()
                
                print(f"   Max probability: {max_prob:.3f}")
                print(f"   Average probability: {avg_prob:.3f}")
                
                if max_prob > 0.9:
                    print("   ⚠️  Very high confidence - model might be overconfident")
                elif max_prob < 0.3:
                    print("   ⚠️  Low confidence - model might be uncertain")
                else:
                    print("   ✅ Reasonable confidence level")
                    
        except Exception as e:
            print(f"   Error checking confidence: {e}")

if __name__ == "__main__":
    import torch
    debug_model_loading()
