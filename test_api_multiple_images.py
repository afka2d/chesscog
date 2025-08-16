#!/usr/bin/env python3
"""
Test the API with multiple images to show current performance.
"""

import requests
import json
import time
from pathlib import Path
import glob

def test_api_with_image(image_path, annotation_path):
    """Test the API with a specific image and its annotation."""
    
    # Load annotation to get corners and expected FEN
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    corners = annotation['corners']
    expected_fen = annotation['fen']
    
    # API endpoint
    api_url = "http://159.203.102.249:8000/recognize_with_manual_corners"
    
    # Prepare the request
    files = {
        'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    try:
        # Make the request
        start_time = time.time()
        response = requests.post(api_url, files=files, data=data, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            actual_fen = result.get('fen', 'N/A')
            pieces_found = result.get('pieces_found', 0)
            processing_time = end_time - start_time
            
            print(f"Image: {Path(image_path).name}")
            print(f"Expected: {expected_fen}")
            print(f"Actual:   {actual_fen}")
            print(f"Pieces:   {pieces_found}")
            print(f"Time:     {processing_time:.2f}s")
            print(f"Match:    {'✅' if actual_fen == expected_fen else '❌'}")
            print("-" * 60)
            
            return actual_fen == expected_fen, pieces_found, processing_time
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False, 0, 0
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, 0, 0

def main():
    """Test multiple images."""
    print("=== Testing API with Multiple Images ===")
    print()
    
    # Find test images with annotations
    test_images = glob.glob("grey_background_dataset/images/test/*.JPG")[:5]  # Test first 5
    
    correct_count = 0
    total_count = 0
    total_pieces = 0
    total_time = 0
    
    for image_path in test_images:
        image_name = Path(image_path).stem
        annotation_path = f"grey_background_dataset/annotations/test/{image_name}.json"
        
        if Path(annotation_path).exists():
            is_correct, pieces, proc_time = test_api_with_image(image_path, annotation_path)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            total_pieces += pieces
            total_time += proc_time
    
    print()
    print("=== SUMMARY ===")
    print(f"Exact FEN matches: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Average pieces detected: {total_pieces/total_count:.1f}")
    print(f"Average processing time: {total_time/total_count:.2f}s")
    print()
    print("💡 Note: Even if FEN doesn't match exactly, the occupancy")
    print("   detection might be good and piece types partially correct.")

if __name__ == "__main__":
    main()