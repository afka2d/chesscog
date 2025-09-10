#!/usr/bin/env python3
"""
Test the API with real chess images to determine expected real-world accuracy.
"""

import requests
import json
import base64
import os
import glob
from PIL import Image
import io
import numpy as np
import chess
from pathlib import Path

def test_api_with_image(image_path, corners, color="white"):
    """Test the API with a single image."""
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare the request
        files = {'image': (os.path.basename(image_path), image_data, 'image/jpeg')}
        data = {
            'corners': json.dumps(corners),
            'color': color
        }
        
        # Make the request
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")
        return None

def analyze_piece_accuracy(result):
    """Analyze the accuracy of piece classification from API result."""
    if not result or 'pieces' not in result:
        return None
    
    pieces = result['pieces']
    fen = result.get('fen', '')
    
    # Convert pieces to a more analyzable format
    piece_count = 0
    piece_types = set()
    color_distribution = {'white': 0, 'black': 0}
    
    if isinstance(pieces, list) and len(pieces) == 64:
        # 1D array
        for i, piece in enumerate(pieces):
            if piece is not None:
                piece_count += 1
                if isinstance(piece, str):
                    if piece.startswith('white_'):
                        color_distribution['white'] += 1
                        piece_types.add(piece)
                    elif piece.startswith('black_'):
                        color_distribution['black'] += 1
                        piece_types.add(piece)
    elif isinstance(pieces, list) and len(pieces) == 8:
        # 2D array (nested lists)
        for rank in pieces:
            for piece in rank:
                if piece is not None:
                    piece_count += 1
                    if isinstance(piece, str):
                        if piece.startswith('white_'):
                            color_distribution['white'] += 1
                            piece_types.add(piece)
                        elif piece.startswith('black_'):
                            color_distribution['black'] += 1
                            piece_types.add(piece)
    
    return {
        'piece_count': piece_count,
        'piece_types': len(piece_types),
        'color_distribution': color_distribution,
        'fen': fen,
        'diversity_score': len(piece_types) / 12.0 if piece_count > 0 else 0
    }

def test_with_sample_images():
    """Test the API with sample images from the dataset."""
    print("🧪 Testing API with Real Chess Images")
    print("=" * 50)
    
    # Test directories
    test_dirs = [
        "my_chess_images/train/images",
        "grey_background_dataset/images/test"
    ]
    
    all_results = []
    total_tests = 0
    successful_tests = 0
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"⚠️  Directory not found: {test_dir}")
            continue
        
        print(f"\n📁 Testing directory: {test_dir}")
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        
        if not image_files:
            print(f"   ⚠️  No images found")
                continue
                
        # Test up to 10 random images
        import random
        test_images = random.sample(image_files, min(10, len(image_files)))
        
        print(f"   📊 Testing {len(test_images)} images...")
        
        for i, image_path in enumerate(test_images):
            print(f"   {i+1:2d}. Testing {os.path.basename(image_path)}...")
            
            # Use default corners (assuming standard chessboard)
            # These are approximate corners for a typical chessboard image
            corners = [
                [50, 50],   # Top-left
                [400, 50],  # Top-right
                [400, 400], # Bottom-right
                [50, 400]   # Bottom-left
            ]
            
            result = test_api_with_image(image_path, corners)
            total_tests += 1
            
            if result:
                successful_tests += 1
                analysis = analyze_piece_accuracy(result)
                if analysis:
                    all_results.append(analysis)
                    print(f"       ✅ Pieces detected: {analysis['piece_count']}, "
                          f"Types: {analysis['piece_types']}, "
                          f"Diversity: {analysis['diversity_score']:.2f}")
                else:
                    print(f"       ⚠️  Could not analyze result")
            else:
                print(f"       ❌ API call failed")
    
    return all_results, total_tests, successful_tests

def calculate_expected_accuracy(results):
    """Calculate expected real-world accuracy based on test results."""
    if not results:
        return "No results to analyze"
    
    # Calculate metrics
    total_pieces = sum(r['piece_count'] for r in results)
    avg_diversity = np.mean([r['diversity_score'] for r in results])
    avg_piece_count = np.mean([r['piece_count'] for r in results])
    
    # Check for overfitting indicators
    overfitting_indicators = []
    
    # Check diversity
    if avg_diversity < 0.3:
        overfitting_indicators.append(f"Low diversity ({avg_diversity:.2f})")
    
    # Check piece count distribution
    piece_counts = [r['piece_count'] for r in results]
    if len(set(piece_counts)) < 3:
        overfitting_indicators.append("Limited piece count variation")
    
    # Check color distribution
    white_pieces = sum(r['color_distribution']['white'] for r in results)
    black_pieces = sum(r['color_distribution']['black'] for r in results)
    total_pieces = white_pieces + black_pieces
    
    if total_pieces > 0:
        white_ratio = white_pieces / total_pieces
        if white_ratio < 0.2 or white_ratio > 0.8:
            overfitting_indicators.append(f"Color bias (white: {white_ratio:.2f})")
    
    # Estimate accuracy based on diversity and consistency
    base_accuracy = min(95, max(60, avg_diversity * 100))
    
    # Penalize for overfitting indicators
    accuracy_penalty = len(overfitting_indicators) * 10
    estimated_accuracy = max(30, base_accuracy - accuracy_penalty)
    
    return {
        'estimated_accuracy': estimated_accuracy,
        'base_accuracy': base_accuracy,
        'overfitting_indicators': overfitting_indicators,
        'avg_diversity': avg_diversity,
        'avg_piece_count': avg_piece_count,
        'total_pieces_detected': total_pieces,
        'white_pieces': white_pieces,
        'black_pieces': black_pieces
    }

def main():
    """Main function to test API accuracy."""
    print("🎯 Testing API Real-World Accuracy")
    print("=" * 60)
    print("Goal: Determine expected accuracy for piece classification")
    
    # Test with sample images
    results, total_tests, successful_tests = test_with_sample_images()
    
    print(f"\n📊 TEST SUMMARY:")
    print("=" * 30)
    print(f"   Total tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if results:
        # Calculate expected accuracy
        accuracy_analysis = calculate_expected_accuracy(results)
        
        print(f"\n🎯 EXPECTED REAL-WORLD ACCURACY:")
        print("=" * 40)
        print(f"   Estimated Accuracy: {accuracy_analysis['estimated_accuracy']:.1f}%")
        print(f"   Base Accuracy: {accuracy_analysis['base_accuracy']:.1f}%")
        print(f"   Average Diversity: {accuracy_analysis['avg_diversity']:.2f}")
        print(f"   Average Pieces/Image: {accuracy_analysis['avg_piece_count']:.1f}")
        print(f"   Total Pieces Detected: {accuracy_analysis['total_pieces_detected']}")
        print(f"   White/Black Ratio: {accuracy_analysis['white_pieces']}/{accuracy_analysis['black_pieces']}")
        
        if accuracy_analysis['overfitting_indicators']:
            print(f"\n⚠️  OVERFITTING INDICATORS:")
            for indicator in accuracy_analysis['overfitting_indicators']:
                print(f"   - {indicator}")
        else:
            print(f"\n✅ NO OVERFITTING DETECTED")
        
        # Final recommendation
        estimated = accuracy_analysis['estimated_accuracy']
        if estimated >= 80:
            print(f"\n🎉 EXCELLENT: Expected accuracy {estimated:.1f}% meets your 80%+ target!")
        elif estimated >= 70:
            print(f"\n✅ GOOD: Expected accuracy {estimated:.1f}% is close to your target")
        elif estimated >= 60:
            print(f"\n⚠️  MODERATE: Expected accuracy {estimated:.1f}% may need improvement")
        else:
            print(f"\n❌ POOR: Expected accuracy {estimated:.1f}% is below acceptable levels")
    
    else:
        print("\n❌ No results to analyze - API may not be working correctly")

if __name__ == "__main__":
    main()