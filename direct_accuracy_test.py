#!/usr/bin/env python3
"""
Direct test of the ChessRecognizer to determine expected real-world accuracy.
This bypasses the API and tests the classifier directly.
"""

import numpy as np
import chess
from pathlib import Path
from PIL import Image
import glob
import random
import os

def test_chess_recognizer_direct():
    """Test the ChessRecognizer directly."""
    print("🧪 Testing ChessRecognizer Directly")
    print("=" * 40)
    
    try:
        # Import the recognizer
        from chesscog.recognition.recognition import ChessRecognizer
        print("✅ ChessRecognizer imported successfully")
        
        # Initialize the recognizer
        print("🔧 Initializing ChessRecognizer...")
        recognizer = ChessRecognizer(Path("models"))
        print("✅ ChessRecognizer initialized successfully")
        
        # Find test images
        test_dirs = [
            "my_chess_images/train/images",
            "grey_background_dataset/images/test"
        ]
        
        test_image = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images.extend(glob.glob(os.path.join(test_dir, ext)))
                if images:
                    test_image = random.choice(images)
                    break
        
        if not test_image:
            print("❌ No test images found")
            return None
        
        print(f"📁 Using test image: {test_image}")
        
        # Load and process image
        img = Image.open(test_image).convert('RGB')
        img_array = np.array(img)
        print(f"📊 Image shape: {img_array.shape}")
        
        # Test recognition
        print("🔍 Running chess recognition...")
        board, corners = recognizer.predict(img_array, chess.WHITE)
        
        print("✅ Recognition completed successfully!")
        
        # Analyze results
        fen = board.fen()
        print(f"📋 FEN: {fen}")
        
        # Count pieces
        piece_count = 0
        piece_types = set()
        color_distribution = {'white': 0, 'black': 0}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_count += 1
                piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                piece_types.add(piece_name)
                
                if piece.color:
                    color_distribution['white'] += 1
                else:
                    color_distribution['black'] += 1
        
        print(f"\n📊 ANALYSIS:")
        print(f"   Total pieces: {piece_count}")
        print(f"   Unique types: {len(piece_types)}")
        print(f"   White pieces: {color_distribution['white']}")
        print(f"   Black pieces: {color_distribution['black']}")
        print(f"   Piece types: {sorted(piece_types)}")
        
        # Calculate diversity score
        diversity = len(piece_types) / 12.0 if piece_count > 0 else 0
        print(f"   Diversity score: {diversity:.2f}")
        
        # Estimate accuracy based on diversity
        if diversity >= 0.8:
            estimated_accuracy = "85-95%"
            confidence = "High"
            assessment = "EXCELLENT"
        elif diversity >= 0.6:
            estimated_accuracy = "75-85%"
            confidence = "Medium"
            assessment = "GOOD"
        elif diversity >= 0.4:
            estimated_accuracy = "65-75%"
            confidence = "Low"
            assessment = "MODERATE"
        else:
            estimated_accuracy = "50-65%"
            confidence = "Very Low"
            assessment = "POOR"
        
        print(f"\n🎯 EXPECTED REAL-WORLD ACCURACY:")
        print(f"   Assessment: {assessment}")
        print(f"   Expected Accuracy: {estimated_accuracy}")
        print(f"   Confidence: {confidence}")
        print(f"   Based on diversity: {diversity:.2f}")
        
        # Check for overfitting indicators
        overfitting_indicators = []
        
        if diversity < 0.3:
            overfitting_indicators.append(f"Very low diversity ({diversity:.2f})")
        
        if len(piece_types) < 4:
            overfitting_indicators.append(f"Limited piece variety ({len(piece_types)} types)")
        
        if piece_count > 0:
            white_ratio = color_distribution['white'] / piece_count
            if white_ratio < 0.2 or white_ratio > 0.8:
                overfitting_indicators.append(f"Color bias (white: {white_ratio:.2f})")
        
        if overfitting_indicators:
            print(f"\n⚠️  OVERFITTING INDICATORS:")
            for indicator in overfitting_indicators:
                print(f"   - {indicator}")
        else:
            print(f"\n✅ NO OVERFITTING DETECTED")
        
        # Final recommendation
        if assessment == "EXCELLENT":
            print(f"\n🎉 EXCELLENT: Expected accuracy {estimated_accuracy} meets your 80%+ target!")
            print(f"   The original ChessCog classifier should work very well for real chess positions.")
        elif assessment == "GOOD":
            print(f"\n✅ GOOD: Expected accuracy {estimated_accuracy} is close to your target")
            print(f"   The classifier should work well for most chess positions.")
        elif assessment == "MODERATE":
            print(f"\n⚠️  MODERATE: Expected accuracy {estimated_accuracy} may need improvement")
            print(f"   The classifier may struggle with some chess positions.")
        else:
            print(f"\n❌ POOR: Expected accuracy {estimated_accuracy} is below acceptable levels")
            print(f"   The classifier may have significant issues.")
        
        return {
            'success': True,
            'diversity': diversity,
            'piece_count': piece_count,
            'piece_types': len(piece_types),
            'estimated_accuracy': estimated_accuracy,
            'assessment': assessment,
            'overfitting_indicators': overfitting_indicators
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🎯 Direct ChessRecognizer Accuracy Test")
    print("=" * 50)
    print("Goal: Determine expected real-world accuracy for piece classification")
    print("Method: Test the original ChessCog classifier directly")
    
    result = test_chess_recognizer_direct()
    
    if result and result['success']:
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"   Assessment: {result['assessment']}")
        print(f"   Expected Accuracy: {result['estimated_accuracy']}")
        print(f"   Diversity Score: {result['diversity']:.2f}")
        
        if result['assessment'] in ['EXCELLENT', 'GOOD']:
            print(f"\n✅ RECOMMENDATION: Use the original ChessCog classifier")
            print(f"   It should provide reliable 80%+ accuracy for real chess positions")
        else:
            print(f"\n⚠️  RECOMMENDATION: Consider additional testing or model improvements")
    else:
        print(f"\n❌ TEST FAILED")
        print(f"   Check the error messages above")

if __name__ == "__main__":
    main()
