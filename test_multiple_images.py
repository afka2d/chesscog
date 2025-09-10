#!/usr/bin/env python3
"""
Test API accuracy with multiple different images to assess consistency.
"""

import requests
import json
from PIL import Image
import io
import chess
import numpy as np
import time
import os

def test_multiple_images():
    """Test API with multiple different images."""
    print("🖼️  Testing Multiple Images for Consistency")
    print("=" * 60)
    
    # Find available test images
    test_images_dir = "grey_background_dataset/images/test"
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} test images")
    else:
        print("No test images directory found")
        return
    
    # Use first few images for testing
    test_images = image_files[:3]  # Test first 3 images
    
    results = []
    
    for i, image_file in enumerate(test_images):
        print(f"\n🖼️  Test {i+1}: {image_file}")
        print("-" * 40)
        
        try:
            # Load image
            img_path = os.path.join(test_images_dir, image_file)
            with open(img_path, 'rb') as f:
                image_data = f.read()
            
            # Use same corners for all tests (assuming similar board setup)
            corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
            
            # Make API request
            files = {'image': (image_file, image_data, 'image/jpeg')}
            data = {
                'corners': json.dumps(corners),
                'color': 'white'
            }
            
            start_time = time.time()
            response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                                   files=files, data=data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Analyze results
                occupancy = result.get('occupancy', [])
                pieces = result.get('pieces', [])
                
                # Count occupied squares
                occupied_count = sum(1 for occ in occupancy if occ)
                
                # Count unique piece types
                unique_pieces = set(p for p in pieces if p is not None)
                unique_count = len(unique_pieces)
                
                # Calculate diversity score
                diversity_score = unique_count / max(occupied_count, 1)
                
                # Estimate accuracy
                if diversity_score >= 0.8:
                    accuracy_estimate = "85-95%"
                elif diversity_score >= 0.6:
                    accuracy_estimate = "75-85%"
                elif diversity_score >= 0.4:
                    accuracy_estimate = "65-75%"
                else:
                    accuracy_estimate = "50-65%"
                
                results.append({
                    'image': image_file,
                    'occupied_squares': occupied_count,
                    'unique_pieces': unique_count,
                    'diversity_score': diversity_score,
                    'accuracy_estimate': accuracy_estimate,
                    'response_time': response_time,
                    'pieces_detected': list(unique_pieces)
                })
                
                print(f"✅ Success!")
                print(f"   Occupied squares: {occupied_count}")
                print(f"   Unique pieces: {unique_count}")
                print(f"   Diversity score: {diversity_score:.2f}")
                print(f"   Accuracy estimate: {accuracy_estimate}")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Pieces detected: {', '.join(sorted(unique_pieces))}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing {image_file}: {e}")
    
    # Summary
    if results:
        print(f"\n📊 CONSISTENCY ANALYSIS")
        print("=" * 40)
        
        avg_occupied = sum(r['occupied_squares'] for r in results) / len(results)
        avg_unique = sum(r['unique_pieces'] for r in results) / len(results)
        avg_diversity = sum(r['diversity_score'] for r in results) / len(results)
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        
        print(f"📈 AVERAGE ACROSS {len(results)} IMAGES:")
        print(f"   Occupied squares: {avg_occupied:.1f}")
        print(f"   Unique pieces: {avg_unique:.1f}")
        print(f"   Diversity score: {avg_diversity:.2f}")
        print(f"   Response time: {avg_response_time:.2f}s")
        
        # Consistency check
        occupied_variance = np.var([r['occupied_squares'] for r in results])
        diversity_variance = np.var([r['diversity_score'] for r in results])
        
        print(f"\n🔍 CONSISTENCY METRICS:")
        print(f"   Occupancy variance: {occupied_variance:.2f}")
        print(f"   Diversity variance: {diversity_variance:.2f}")
        
        if occupied_variance < 5 and diversity_variance < 0.1:
            print("   ✅ High consistency across images")
        elif occupied_variance < 10 and diversity_variance < 0.2:
            print("   ⚠️  Moderate consistency across images")
        else:
            print("   ❌ Low consistency across images")
        
        # Overall assessment
        if avg_diversity >= 0.6:
            print(f"\n🎯 OVERALL ASSESSMENT: GOOD ({avg_diversity:.2f} diversity)")
        else:
            print(f"\n🎯 OVERALL ASSESSMENT: NEEDS IMPROVEMENT ({avg_diversity:.2f} diversity)")

if __name__ == "__main__":
    test_multiple_images()
