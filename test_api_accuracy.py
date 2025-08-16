#!/usr/bin/env python3
"""
Test script to evaluate the API's accuracy across multiple test images.
Compares API predictions with ground truth annotations.
"""

import os
import json
import requests
from pathlib import Path
import chess
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

def load_ground_truth(json_path: str) -> Tuple[str, List[List[float]]]:
    """Load ground truth FEN and corners from annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data['fen'], data['corners']

def compare_positions(true_fen: str, pred_fen: str) -> Dict[str, float]:
    """
    Compare two chess positions and return detailed metrics.
    Returns:
        - piece_accuracy: % of squares with correct piece type and color
        - occupancy_accuracy: % of squares with correct occupancy (piece/empty)
        - piece_type_accuracy: % of occupied squares with correct piece type
    """
    true_board = chess.Board(true_fen)
    pred_board = chess.Board(pred_fen)
    
    total_squares = 64
    correct_pieces = 0
    correct_occupancy = 0
    correct_piece_types = 0
    occupied_squares = 0
    
    for square in chess.SQUARES:
        true_piece = true_board.piece_at(square)
        pred_piece = pred_board.piece_at(square)
        
        # Check occupancy
        if (true_piece is None) == (pred_piece is None):
            correct_occupancy += 1
            
        if true_piece is not None:
            occupied_squares += 1
            if pred_piece is not None:
                # Check piece type
                if true_piece.piece_type == pred_piece.piece_type:
                    correct_piece_types += 1
                # Check complete piece match (type and color)
                if true_piece == pred_piece:
                    correct_pieces += 1
    
    return {
        'piece_accuracy': correct_pieces / total_squares,
        'occupancy_accuracy': correct_occupancy / total_squares,
        'piece_type_accuracy': correct_piece_types / occupied_squares if occupied_squares > 0 else 1.0
    }

def test_api_accuracy(test_dir: str = 'grey_background_dataset', api_url: str = 'http://159.203.102.249:8000'):
    """Test API accuracy on all test images."""
    
    # Paths
    test_images_dir = Path(test_dir) / 'images' / 'test'
    test_annotations_dir = Path(test_dir) / 'annotations' / 'test'
    
    # Collect results
    results = defaultdict(list)
    errors = []
    
    # Process each test image
    test_files = list(test_images_dir.glob('*.JPG'))
    print(f"Found {len(test_files)} test images")
    
    for img_path in tqdm(test_files, desc="Testing images"):
        json_path = test_annotations_dir / f"{img_path.stem}.json"
        
        try:
            # Load ground truth
            true_fen, corners = load_ground_truth(str(json_path))
            
            # Call API
            with open(img_path, 'rb') as img_file:
                response = requests.post(
                    f"{api_url}/recognize_with_manual_corners",
                    files={'image': ('image.jpg', img_file, 'image/jpeg')},
                    data={'corners': json.dumps(corners)}
                )
            
            if response.status_code != 200:
                errors.append(f"API error for {img_path.name}: {response.status_code}")
                continue
                
            # Get API prediction
            api_result = response.json()
            pred_fen = api_result['fen']
            
            # Compare positions
            metrics = compare_positions(true_fen, pred_fen)
            
            # Store results
            for metric, value in metrics.items():
                results[metric].append(value)
                
            # Store detailed results for this image
            results['details'].append({
                'image': img_path.name,
                'true_fen': true_fen,
                'pred_fen': pred_fen,
                'pieces_found': api_result.get('pieces_found', 0),
                **metrics
            })
            
        except Exception as e:
            errors.append(f"Error processing {img_path.name}: {str(e)}")
    
    # Calculate overall metrics
    overall_metrics = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        for metric, values in results.items()
        if metric != 'details'
    }
    
    # Save results
    output = {
        'overall_metrics': overall_metrics,
        'image_details': results['details'],
        'errors': errors
    }
    
    output_path = Path('api_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\nTest Results Summary:")
    print("-" * 50)
    for metric, stats in overall_metrics.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {stats['mean']:.2%}")
        print(f"  Std:  {stats['std']:.2%}")
        print(f"  Min:  {stats['min']:.2%}")
        print(f"  Max:  {stats['max']:.2%}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    test_api_accuracy()