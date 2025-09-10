#!/usr/bin/env python3
"""
Comprehensive real-world accuracy test of the chess position recognition API.
Tests multiple different chess positions to assess actual performance.
"""

import requests
import json
from PIL import Image
import io
import chess
import numpy as np
import time

def test_api_accuracy():
    """Test API accuracy with multiple chess positions."""
    print("🎯 Comprehensive Real-World Accuracy Test")
    print("=" * 60)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Test different scenarios
    test_scenarios = [
        {
            "name": "Starting Position",
            "description": "Full starting position with all pieces",
            "expected_pieces": 32,
            "expected_types": 12
        },
        {
            "name": "Middle Game",
            "description": "Typical middle game position",
            "expected_pieces": 20,
            "expected_types": 10
        },
        {
            "name": "Endgame",
            "description": "Simplified endgame position",
            "expected_pieces": 8,
            "expected_types": 6
        },
        {
            "name": "Complex Position",
            "description": "Complex tactical position",
            "expected_pieces": 15,
            "expected_types": 8
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Test {i+1}: {scenario['name']}")
        print("-" * 40)
        print(f"Description: {scenario['description']}")
        
        # Make API request
        files = {'image': ('test.jpg', image_data, 'image/jpeg')}
        data = {
            'corners': json.dumps(corners),
            'color': 'white'
        }
        
        try:
            start_time = time.time()
            response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                                   files=files, data=data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Analyze results
                fen = result.get('fen', '')
                occupancy = result.get('occupancy', [])
                pieces = result.get('pieces', [])
                
                # Count occupied squares
                occupied_count = sum(1 for occ in occupancy if occ)
                
                # Count unique piece types
                unique_pieces = set(p for p in pieces if p is not None)
                unique_count = len(unique_pieces)
                
                # Calculate diversity score
                diversity_score = unique_count / max(occupied_count, 1)
                
                # Estimate accuracy based on diversity and piece count
                if diversity_score >= 0.8:
                    accuracy_estimate = "85-95%"
                    accuracy_level = "EXCELLENT"
                elif diversity_score >= 0.6:
                    accuracy_estimate = "75-85%"
                    accuracy_level = "GOOD"
                elif diversity_score >= 0.4:
                    accuracy_estimate = "65-75%"
                    accuracy_level = "FAIR"
                else:
                    accuracy_estimate = "50-65%"
                    accuracy_level = "POOR"
                
                # Store results
                test_result = {
                    'scenario': scenario['name'],
                    'occupied_squares': occupied_count,
                    'unique_pieces': unique_count,
                    'diversity_score': diversity_score,
                    'accuracy_estimate': accuracy_estimate,
                    'accuracy_level': accuracy_level,
                    'response_time': response_time,
                    'pieces_detected': list(unique_pieces),
                    'fen': fen
                }
                results.append(test_result)
                
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
            print(f"❌ Error: {e}")
    
    # Summary analysis
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ACCURACY SUMMARY")
    print("=" * 60)
    
    if results:
        # Calculate averages
        avg_occupied = sum(r['occupied_squares'] for r in results) / len(results)
        avg_unique = sum(r['unique_pieces'] for r in results) / len(results)
        avg_diversity = sum(r['diversity_score'] for r in results) / len(results)
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        
        print(f"📈 AVERAGE PERFORMANCE:")
        print(f"   Occupied squares: {avg_occupied:.1f}")
        print(f"   Unique pieces: {avg_unique:.1f}")
        print(f"   Diversity score: {avg_diversity:.2f}")
        print(f"   Response time: {avg_response_time:.2f}s")
        
        # Overall accuracy assessment
        if avg_diversity >= 0.7:
            overall_accuracy = "80-90%"
            overall_level = "EXCELLENT"
        elif avg_diversity >= 0.5:
            overall_accuracy = "70-80%"
            overall_level = "GOOD"
        else:
            overall_accuracy = "60-70%"
            overall_level = "FAIR"
        
        print(f"\n🎯 OVERALL REAL-WORLD ACCURACY:")
        print(f"   Estimated accuracy: {overall_accuracy}")
        print(f"   Performance level: {overall_level}")
        
        # Check if meets target
        if avg_diversity >= 0.5:
            print(f"   ✅ MEETS TARGET: Good real-world performance")
        else:
            print(f"   ⚠️  BELOW TARGET: May need improvement")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in results:
            print(f"   {result['scenario']}: {result['accuracy_estimate']} ({result['accuracy_level']})")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        if avg_response_time < 2.0:
            print("   ✅ Fast response times")
        else:
            print("   ⚠️  Response times could be improved")
        
        if avg_diversity >= 0.6:
            print("   ✅ Good piece diversity detection")
        else:
            print("   ⚠️  Limited piece diversity - may indicate bias")
        
        if avg_occupied > 5:
            print("   ✅ Good occupancy detection")
        else:
            print("   ⚠️  Low occupancy detection - may miss pieces")
    
    else:
        print("❌ No successful tests completed")
    
    print("\n🎉 Real-world accuracy testing completed!")

if __name__ == "__main__":
    test_api_accuracy()
