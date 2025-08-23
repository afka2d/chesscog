#!/usr/bin/env python3
"""
Test the two-stage classifier on completely unseen validation data.
This will give us a realistic assessment of performance.
"""

import os
import numpy as np
from PIL import Image
import chess
from two_stage_piece_classifier import TwoStagePieceClassifier

def test_classifier_on_validation():
    """Test the classifier on validation set (unseen during training)."""
    
    print("🧪 Testing Two-Stage Classifier on VALIDATION SET (Unseen Data)")
    print("=" * 60)
    
    # Load the trained classifier
    try:
        classifier = TwoStagePieceClassifier()
        print("✅ Classifier loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return
    
    # Test on validation set (completely unseen during training)
    val_dir = 'grey_background_dataset/pieces/val'
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    total_correct = 0
    total_samples = 0
    class_results = {}
    
    # Test each piece class
    for piece_type in sorted(os.listdir(val_dir)):
        piece_dir = os.path.join(val_dir, piece_type)
        if not os.path.isdir(piece_dir):
            continue
            
        print(f"\n📁 Testing {piece_type}:")
        correct = 0
        samples = 0
        
        # Get all PNG files in this class
        png_files = [f for f in os.listdir(piece_dir) if f.endswith('.png')]
        
        # Test first 20 images per class (to avoid overwhelming output)
        test_files = png_files[:20]
        
        for img_file in test_files:
            img_path = os.path.join(piece_dir, img_file)
            try:
                # Load and convert image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Classify the piece
                result = classifier.classify_piece(img_array)
                
                if len(result) == 4:
                    piece_name, confidence, color_conf, piece_conf = result
                else:
                    print(f"  ❌ Unexpected result format: {result}")
                    continue
                
                # Check if prediction matches the directory name
                if piece_name == piece_type:
                    correct += 1
                    total_correct += 1
                    print(f"  ✅ {img_file}: {piece_name} (conf: {confidence:.3f})")
                else:
                    print(f"  ❌ {img_file}: predicted {piece_name}, actual {piece_type} (conf: {confidence:.3f})")
                
                samples += 1
                total_samples += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
        
        # Calculate accuracy for this class
        if samples > 0:
            accuracy = (correct / samples) * 100
            class_results[piece_type] = accuracy
            print(f"  📊 {correct}/{samples} correct ({accuracy:.1f}%)")
        else:
            class_results[piece_type] = 0
            print(f"  📊 No valid samples")
    
    # Overall results
    print(f"\n🎯 OVERALL VALIDATION RESULTS")
    print("=" * 40)
    print(f"📊 Total samples tested: {total_samples}")
    print(f"✅ Total correct: {total_correct}")
    
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        print(f"🎯 Overall accuracy: {overall_accuracy:.1f}%")
        
        # Class-by-class breakdown
        print(f"\n📈 Per-Class Accuracy:")
        for piece_type, accuracy in sorted(class_results.items()):
            print(f"   {piece_type}: {accuracy:.1f}%")
        
        # Compare with training accuracy
        print(f"\n🔍 Analysis:")
        if overall_accuracy > 95:
            print("   ⚠️  Very high accuracy - potential overfitting")
        elif overall_accuracy > 85:
            print("   ✅ Good accuracy - reasonable performance")
        elif overall_accuracy > 70:
            print("   ⚠️  Moderate accuracy - some overfitting possible")
        else:
            print("   ❌ Low accuracy - significant overfitting or training issues")
    else:
        print("❌ No valid samples were tested")

if __name__ == "__main__":
    test_classifier_on_validation()

