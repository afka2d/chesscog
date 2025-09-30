#!/usr/bin/env python3
"""
Compare the original API vs Marshall Improved API.
Tests both APIs with the same image to compare results.
"""

import requests
import json
import time
from pathlib import Path

def test_api(api_url, api_name, sample_image, corners):
    """Test a specific API endpoint"""
    try:
        print(f"\n🧪 Testing {api_name}...")
        print(f"   URL: {api_url}")
        
        # Prepare the request
        files = {'image': open(sample_image, 'rb')}
        data = {
            'corners': json.dumps(corners),
            'debug': 'true'
        }
        
        # Make the request
        start_time = time.time()
        response = requests.post(
            f'{api_url}/recognize_chess_position_with_corners',
            files=files,
            data=data,
            timeout=30
        )
        processing_time = time.time() - start_time
        
        files['image'].close()
        
        if response.status_code == 200:
            result = response.json()
            pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
            
            print(f"   ✅ Success")
            print(f"   FEN: {result.get('fen', 'N/A')}")
            print(f"   Pieces detected: {pieces_detected}")
            print(f"   Processing time: {processing_time:.3f}s")
            
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"   Occupied squares: {debug.get('occupied_squares', 0)}")
                print(f"   Squares processed: {debug.get('squares_processed', 0)}")
            
            return {
                'success': True,
                'fen': result.get('fen'),
                'pieces_detected': pieces_detected,
                'processing_time': processing_time,
                'pieces': result.get('pieces', []),
                'occupancy': result.get('occupancy', []),
                'debug_info': result.get('debug_info', {})
            }
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def compare_results(original_result, marshall_result):
    """Compare the results from both APIs"""
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    if not original_result['success']:
        print("❌ Original API failed - cannot compare")
        return
    
    if not marshall_result['success']:
        print("❌ Marshall API failed - cannot compare")
        return
    
    print(f"FEN Comparison:")
    print(f"   Original:  {original_result['fen']}")
    print(f"   Marshall:  {marshall_result['fen']}")
    print(f"   Match:     {'✅ YES' if original_result['fen'] == marshall_result['fen'] else '❌ NO'}")
    
    print(f"\nPieces Detected:")
    print(f"   Original:  {original_result['pieces_detected']}")
    print(f"   Marshall:  {marshall_result['pieces_detected']}")
    print(f"   Difference: {marshall_result['pieces_detected'] - original_result['pieces_detected']:+d}")
    
    print(f"\nProcessing Time:")
    print(f"   Original:  {original_result['processing_time']:.3f}s")
    print(f"   Marshall:  {marshall_result['processing_time']:.3f}s")
    print(f"   Difference: {marshall_result['processing_time'] - original_result['processing_time']:+.3f}s")
    
    # Compare piece by piece
    print(f"\nPiece-by-Piece Comparison:")
    original_pieces = original_result['pieces']
    marshall_pieces = marshall_result['pieces']
    
    matches = 0
    total_occupied = 0
    
    for i, (orig, marsh) in enumerate(zip(original_pieces, marshall_pieces)):
        if orig is not None or marsh is not None:
            total_occupied += 1
            if orig == marsh:
                matches += 1
            else:
                rank = 8 - (i // 8)
                file = chr(97 + (i % 8))
                print(f"   {file}{rank}: Original={orig}, Marshall={marsh}")
    
    if total_occupied > 0:
        accuracy = (matches / total_occupied) * 100
        print(f"\nPiece Classification Accuracy: {accuracy:.1f}% ({matches}/{total_occupied})")
    
    # Compare occupancy
    print(f"\nOccupancy Comparison:")
    original_occupancy = original_result['occupancy']
    marshall_occupancy = marshall_result['occupancy']
    
    occupancy_matches = sum(1 for orig, marsh in zip(original_occupancy, marshall_occupancy) if orig == marsh)
    occupancy_accuracy = (occupancy_matches / 64) * 100
    print(f"   Occupancy Accuracy: {occupancy_accuracy:.1f}% ({occupancy_matches}/64)")

def main():
    print("🔄 Comparing Original API vs Marshall Improved API")
    print("=" * 60)
    
    # API URLs
    original_url = "http://localhost:8001"  # Local development API
    marshall_url = "http://localhost:8003"  # Marshall improved API
    
    # Look for a sample chess image
    sample_images = [
        "data/occupancy/test/occupied/IMG_4767_a8.png",
        "data/occupancy/test/occupied/IMG_4764_c2.png", 
        "data/occupancy/test/occupied/IMG_4763_d7.png"
    ]
    
    sample_image = None
    for img_path in sample_images:
        if Path(img_path).exists():
            sample_image = img_path
            break
    
    if not sample_image:
        print("❌ No sample images found for testing")
        print("   Please ensure you have chess images in the data/occupancy/test/occupied/ directory")
        return 1
    
    print(f"📸 Using sample image: {sample_image}")
    
    # Sample corners (you may need to adjust these based on your image)
    corners = [[324, 324], [2916, 324], [2916, 5436], [324, 5436]]
    
    # Test both APIs
    print("\n🔄 Testing both APIs...")
    
    original_result = test_api(original_url, "Original API (Port 8001)", sample_image, corners)
    marshall_result = test_api(marshall_url, "Marshall Improved API (Port 8003)", sample_image, corners)
    
    # Compare results
    compare_results(original_result, marshall_result)
    
    print("\n" + "=" * 60)
    print("✅ Comparison completed!")
    print("📍 Original API: http://localhost:8001")
    print("📍 Marshall API: http://localhost:8003")
    
    return 0

if __name__ == "__main__":
    exit(main())