#!/usr/bin/env python3
"""
Quick API comparison script - tests 10 images from each dataset.
"""

import requests
import json
import time
import random
from pathlib import Path
import numpy as np
import chess

def load_grey_background_test_data(num_images=10):
    """Load test images from grey background dataset with their annotations."""
    test_images_dir = Path("grey_background_dataset/images/test")
    test_annotations_dir = Path("grey_background_dataset/annotations/test")
    
    if not test_images_dir.exists():
        print(f"❌ Grey background test images not found at {test_images_dir}")
        return []
    
    # Get all available test images
    available_images = list(test_images_dir.glob("*.JPG")) + list(test_images_dir.glob("*.jpg"))
    print(f"📸 Found {len(available_images)} grey background test images")
    
    # Select random subset
    selected_images = random.sample(available_images, min(num_images, len(available_images)))
    
    test_data = []
    for img_path in selected_images:
        # Look for corresponding annotation
        annotation_path = test_annotations_dir / f"{img_path.stem}.json"
        
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                
                # Extract corners and FEN
                corners = annotation.get('corners', [])
                fen = annotation.get('fen', '')
                
                if corners and fen:
                    test_data.append({
                        'image_path': str(img_path),
                        'corners': corners,
                        'fen': fen,
                        'dataset': 'grey_background'
                    })
            except Exception as e:
                print(f"⚠️  Error loading annotation for {img_path.name}: {e}")
                continue
        else:
            print(f"⚠️  No annotation found for {img_path.name}")
            continue
    
    print(f"✅ Loaded {len(test_data)} grey background test samples with annotations")
    return test_data

def test_api_on_images(api_url, api_name, test_data, timeout=30):
    """Test an API on a list of test images."""
    print(f"\n🧪 Testing {api_name} on {len(test_data)} images...")
    print(f"   URL: {api_url}")
    
    results = []
    successful_tests = 0
    failed_tests = 0
    
    for i, test_sample in enumerate(test_data):
        try:
            print(f"   [{i+1}/{len(test_data)}] Testing {Path(test_sample['image_path']).name}...")
            
            # Prepare the request
            files = {'image': open(test_sample['image_path'], 'rb')}
            data = {
                'corners': json.dumps(test_sample['corners']),
                'debug': 'true'
            }
            
            # Make the request
            start_time = time.time()
            response = requests.post(
                f'{api_url}/recognize_chess_position_with_corners',
                files=files,
                data=data,
                timeout=timeout
            )
            processing_time = time.time() - start_time
            
            files['image'].close()
            
            if response.status_code == 200:
                result = response.json()
                pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
                
                # Calculate accuracy metrics
                predicted_fen = result.get('fen', '')
                true_fen = test_sample['fen']
                
                # Parse FENs to compare board positions
                try:
                    true_board = chess.Board(true_fen)
                    predicted_board = chess.Board(predicted_fen)
                    
                    # Count correct pieces
                    correct_pieces = 0
                    total_pieces = 0
                    
                    for square in chess.SQUARES:
                        true_piece = true_board.piece_at(square)
                        predicted_piece = predicted_board.piece_at(square)
                        
                        if true_piece is not None:
                            total_pieces += 1
                            if true_piece == predicted_piece:
                                correct_pieces += 1
                    
                    accuracy = (correct_pieces / total_pieces * 100) if total_pieces > 0 else 0
                    
                except Exception as e:
                    print(f"     ⚠️  FEN parsing error: {e}")
                    accuracy = 0
                    correct_pieces = 0
                    total_pieces = 1
                
                results.append({
                    'image_name': Path(test_sample['image_path']).name,
                    'success': True,
                    'predicted_fen': predicted_fen,
                    'true_fen': true_fen,
                    'pieces_detected': pieces_detected,
                    'true_pieces': total_pieces,
                    'correct_pieces': correct_pieces,
                    'accuracy': accuracy,
                    'processing_time': processing_time,
                    'dataset': test_sample['dataset']
                })
                
                successful_tests += 1
                print(f"     ✅ Success - Accuracy: {accuracy:.1f}% ({correct_pieces}/{total_pieces}) - Time: {processing_time:.2f}s")
                
            else:
                print(f"     ❌ Failed: HTTP {response.status_code}")
                results.append({
                    'image_name': Path(test_sample['image_path']).name,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'processing_time': processing_time,
                    'dataset': test_sample['dataset']
                })
                failed_tests += 1
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
            results.append({
                'image_name': Path(test_sample['image_path']).name,
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'dataset': test_sample['dataset']
            })
            failed_tests += 1
    
    print(f"   📊 {api_name} Results: {successful_tests} successful, {failed_tests} failed")
    return results

def analyze_results(original_results, marshall_results, dataset_name):
    """Analyze and compare results from both APIs."""
    print(f"\n📊 ANALYSIS: {dataset_name.upper()} DATASET")
    print("=" * 60)
    
    # Filter successful results
    orig_successful = [r for r in original_results if r['success']]
    marshall_successful = [r for r in marshall_results if r['success']]
    
    if not orig_successful or not marshall_successful:
        print("❌ Not enough successful results to compare")
        return
    
    # Calculate metrics
    orig_accuracies = [r['accuracy'] for r in orig_successful]
    marshall_accuracies = [r['accuracy'] for r in marshall_successful]
    
    orig_times = [r['processing_time'] for r in orig_successful]
    marshall_times = [r['processing_time'] for r in marshall_successful]
    
    orig_pieces = [r['pieces_detected'] for r in orig_successful]
    marshall_pieces = [r['pieces_detected'] for r in marshall_successful]
    
    # Print comparison
    print(f"Accuracy Comparison:")
    print(f"   Original API:  {np.mean(orig_accuracies):.1f}% ± {np.std(orig_accuracies):.1f}% (avg ± std)")
    print(f"   Marshall API:  {np.mean(marshall_accuracies):.1f}% ± {np.std(marshall_accuracies):.1f}% (avg ± std)")
    print(f"   Difference:    {np.mean(marshall_accuracies) - np.mean(orig_accuracies):+.1f}%")
    
    print(f"\nProcessing Time Comparison:")
    print(f"   Original API:  {np.mean(orig_times):.3f}s ± {np.std(orig_times):.3f}s")
    print(f"   Marshall API:  {np.mean(marshall_times):.3f}s ± {np.std(marshall_times):.3f}s")
    print(f"   Difference:    {np.mean(marshall_times) - np.mean(orig_times):+.3f}s")
    
    print(f"\nPieces Detected Comparison:")
    print(f"   Original API:  {np.mean(orig_pieces):.1f} ± {np.std(orig_pieces):.1f}")
    print(f"   Marshall API:  {np.mean(marshall_pieces):.1f} ± {np.std(marshall_pieces):.1f}")
    print(f"   Difference:    {np.mean(marshall_pieces) - np.mean(orig_pieces):+.1f}")
    
    # Find best and worst cases
    print(f"\nBest Performance (Highest Accuracy):")
    orig_best = max(orig_successful, key=lambda x: x['accuracy'])
    marshall_best = max(marshall_successful, key=lambda x: x['accuracy'])
    print(f"   Original:  {orig_best['image_name']} - {orig_best['accuracy']:.1f}%")
    print(f"   Marshall:  {marshall_best['image_name']} - {marshall_best['accuracy']:.1f}%")
    
    print(f"\nWorst Performance (Lowest Accuracy):")
    orig_worst = min(orig_successful, key=lambda x: x['accuracy'])
    marshall_worst = min(marshall_successful, key=lambda x: x['accuracy'])
    print(f"   Original:  {orig_worst['image_name']} - {orig_worst['accuracy']:.1f}%")
    print(f"   Marshall:  {marshall_worst['image_name']} - {marshall_worst['accuracy']:.1f}%")

def main():
    print("🔄 Quick API Comparison (10 images)")
    print("=" * 60)
    
    # API URLs
    original_url = "http://localhost:8001"  # Local development API
    marshall_url = "http://localhost:8003"  # Marshall API
    
    # Load test data
    print("\n📸 Loading test data...")
    grey_test_data = load_grey_background_test_data(num_images=10)
    
    if not grey_test_data:
        print("❌ No test data available. Cannot perform comparison.")
        return 1
    
    # Test both APIs on grey background dataset
    print(f"\n🧪 Testing on {len(grey_test_data)} grey background images...")
    
    original_results = test_api_on_images(original_url, "Original API (Port 8001)", grey_test_data)
    marshall_results = test_api_on_images(marshall_url, "Marshall API (Port 8003)", grey_test_data)
    
    # Analyze results
    analyze_results(original_results, marshall_results, "grey_background")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    orig_successful = [r for r in original_results if r['success']]
    marshall_successful = [r for r in marshall_results if r['success']]
    
    if orig_successful and marshall_successful:
        orig_avg_accuracy = np.mean([r['accuracy'] for r in orig_successful])
        marshall_avg_accuracy = np.mean([r['accuracy'] for r in marshall_successful])
        
        print(f"Overall Accuracy:")
        print(f"   Original API:  {orig_avg_accuracy:.1f}%")
        print(f"   Marshall API:  {marshall_avg_accuracy:.1f}%")
        print(f"   Improvement:   {marshall_avg_accuracy - orig_avg_accuracy:+.1f}%")
        
        if marshall_avg_accuracy > orig_avg_accuracy:
            print(f"🎉 Marshall API performs better!")
        elif marshall_avg_accuracy < orig_avg_accuracy:
            print(f"⚠️  Original API performs better")
        else:
            print(f"🤝 Both APIs perform equally")
    
    print(f"\n✅ Comparison completed!")
    print(f"📍 Original API: {original_url}")
    print(f"📍 Marshall API: {marshall_url}")
    
    return 0

if __name__ == "__main__":
    exit(main())
