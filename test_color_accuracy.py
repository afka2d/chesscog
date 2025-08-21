#!/usr/bin/env python3
"""
Test specifically the color classification accuracy of the two-stage classifier.
This will show if the color classifier is working well in the real world.
"""

import os
import numpy as np
from PIL import Image
from two_stage_piece_classifier import TwoStagePieceClassifier

def test_color_accuracy():
    """Test color classification accuracy specifically."""
    
    print("🎨 Testing Color Classification Accuracy on VALIDATION SET")
    print("=" * 60)
    
    # Load the trained classifier
    try:
        classifier = TwoStagePieceClassifier()
        print("✅ Classifier loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return
    
    # Test on validation set
    val_dir = 'grey_background_dataset/pieces/val'
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    total_correct_color = 0
    total_samples = 0
    color_results = {'white': {'correct': 0, 'total': 0}, 'black': {'correct': 0, 'total': 0}}
    
    # Test each piece class
    for piece_type in sorted(os.listdir(val_dir)):
        piece_dir = os.path.join(val_dir, piece_type)
        if not os.path.isdir(piece_dir):
            continue
            
        # Extract color from piece type name
        if piece_type.startswith('white_'):
            actual_color = 'white'
        elif piece_type.startswith('black_'):
            actual_color = 'black'
        else:
            continue
            
        print(f"\n📁 Testing {piece_type} (actual color: {actual_color}):")
        correct_color = 0
        samples = 0
        
        # Get all PNG files in this class
        png_files = [f for f in os.listdir(piece_dir) if f.endswith('.png')]
        test_files = png_files[:20]  # Test first 20 images per class
        
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
                    
                    # Extract predicted color from piece name
                    if piece_name and piece_name != 'None':
                        if piece_name.startswith('white_'):
                            predicted_color = 'white'
                        elif piece_name.startswith('black_'):
                            predicted_color = 'black'
                        else:
                            predicted_color = 'unknown'
                        
                        # Check color accuracy
                        if predicted_color == actual_color:
                            correct_color += 1
                            total_correct_color += 1
                            print(f"  ✅ {img_file}: color correct ({predicted_color}) - conf: {color_conf:.3f}")
                        else:
                            print(f"  ❌ {img_file}: color wrong (predicted {predicted_color}, actual {actual_color}) - conf: {color_conf:.3f}")
                    else:
                        print(f"  ❌ {img_file}: no prediction - color conf: {color_conf:.3f}")
                    
                    samples += 1
                    total_samples += 1
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
        
        # Calculate color accuracy for this class
        if samples > 0:
            color_accuracy = (correct_color / samples) * 100
            color_results[actual_color]['correct'] += correct_color
            color_results[actual_color]['total'] += samples
            print(f"  📊 Color accuracy: {correct_color}/{samples} ({color_accuracy:.1f}%)")
        else:
            print(f"  📊 No valid samples")
    
    # Overall color results
    print(f"\n🎨 OVERALL COLOR CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"📊 Total samples tested: {total_samples}")
    print(f"✅ Total correct colors: {total_correct_color}")
    
    if total_samples > 0:
        overall_color_accuracy = (total_correct_color / total_samples) * 100
        print(f"🎯 Overall color accuracy: {overall_color_accuracy:.1f}%")
        
        # Color-by-color breakdown
        print(f"\n📈 Per-Color Accuracy:")
        for color in ['white', 'black']:
            if color_results[color]['total'] > 0:
                accuracy = (color_results[color]['correct'] / color_results[color]['total']) * 100
                print(f"   {color.capitalize()} pieces: {color_results[color]['correct']}/{color_results[color]['total']} ({accuracy:.1f}%)")
        
        # Analysis
        print(f"\n🔍 Color Classification Analysis:")
        if overall_color_accuracy > 95:
            print("   ✅ Excellent color accuracy - color confusion eliminated!")
        elif overall_color_accuracy > 85:
            print("   ✅ Good color accuracy - minor color confusion")
        elif overall_color_accuracy > 70:
            print("   ⚠️  Moderate color accuracy - some color confusion")
        else:
            print("   ❌ Poor color accuracy - significant color confusion")
            
        # Compare with piece type accuracy
        print(f"\n🎯 Key Finding:")
        if overall_color_accuracy > 70:
            print("   The color classifier IS working well! The main issue is piece type classification.")
            print("   This confirms the two-stage approach is effective for color distinction.")
        else:
            print("   Both color and piece type classification are failing.")
            print("   The two-stage approach needs fundamental improvements.")
    else:
        print("❌ No valid samples were tested")

if __name__ == "__main__":
    test_color_accuracy()
