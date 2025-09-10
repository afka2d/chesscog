#!/usr/bin/env python3
"""
Test script to evaluate the full accuracy of the new API
(occupancy + piece classification) on multiple sample positions.
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path
import chess
import time
from typing import Dict, List, Tuple

# Test positions with known FENs and corner coordinates
TEST_POSITIONS = [
    {
        "name": "Starting Position",
        "image": "grey_background_dataset/images/train/IMG_4679.JPG",
        "expected_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "corners": [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]],
        "turn": "white"
    },
    {
        "name": "Middle Game Position",
        "image": "grey_background_dataset/images/test/IMG_4763.JPG", 
        "expected_fen": "rnbqk2r/1ppp1ppp/5n2/2b5/2BpP3/2N2N2/1PP2PPP/R1BQK2R w KQkq - 0 1",
        "corners": [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]],
        "turn": "white"
    },
    {
        "name": "Endgame Position",
        "image": "grey_background_dataset/images/val/IMG_4754.JPG",
        "expected_fen": "8/8/8/8/8/8/8/8 w - - 0 1",  # Will be updated with actual FEN
        "corners": [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]],
        "turn": "white"
    }
]

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_api_position(position: Dict) -> Dict:
    """Test a single position with the API."""
    print(f"\n🧪 Testing: {position['name']}")
    print(f"   Image: {position['image']}")
    print(f"   Expected FEN: {position['expected_fen']}")
    
    # Check if image exists
    if not Path(position['image']).exists():
        print(f"   ❌ Image not found: {position['image']}")
        return {"error": "Image not found"}
    
    # Prepare API request with form data
    try:
        with open(position['image'], 'rb') as image_file:
            files = {'image': image_file}
            data = {
                'corners': json.dumps(position['corners']),
                'color': position['turn']
            }
            
            # Make API request
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/recognize_chess_position_with_corners",
                files=files,
                data=data,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                predicted_fen = result.get('fen', 'N/A')
                legal = result.get('legal', False)
                pieces_count = result.get('pieces_count', 0)
                
                print(f"   ✅ API Response: {response.status_code}")
                print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
                print(f"   🎯 Predicted FEN: {predicted_fen}")
                print(f"   ⚖️  Legal Position: {legal}")
                print(f"   ♟️  Pieces Count: {pieces_count}")
                
                # Calculate accuracy metrics
                fen_match = (predicted_fen == position['expected_fen'])
                print(f"   🎯 FEN Match: {'✅' if fen_match else '❌'}")
                
                return {
                    "success": True,
                    "predicted_fen": predicted_fen,
                    "expected_fen": position['expected_fen'],
                    "fen_match": fen_match,
                    "legal": legal,
                    "pieces_count": pieces_count,
                    "processing_time": processing_time,
                    "response_time": processing_time
                }
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return {
                    "error": f"API returned {response.status_code}",
                    "response_text": response.text
                }
                
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout (>30s)")
        return {"error": "Request timeout"}
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return {"error": f"Request failed: {e}"}

def analyze_chess_position(fen: str) -> Dict:
    """Analyze a chess position for accuracy metrics."""
    try:
        board = chess.Board(fen)
        
        # Count pieces by type and color
        piece_counts = {
            'white_pawns': 0, 'white_rooks': 0, 'white_knights': 0,
            'white_bishops': 0, 'white_queens': 0, 'white_kings': 0,
            'black_pawns': 0, 'black_rooks': 0, 'black_knights': 0,
            'black_bishops': 0, 'black_queens': 0, 'black_kings': 0
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = 'white' if piece.color == chess.WHITE else 'black'
                piece_type = piece.symbol().lower()
                if piece_type == 'p':
                    piece_counts[f'{color}_pawns'] += 1
                elif piece_type == 'r':
                    piece_counts[f'{color}_rooks'] += 1
                elif piece_type == 'n':
                    piece_counts[f'{color}_knights'] += 1
                elif piece_type == 'b':
                    piece_counts[f'{color}_bishops'] += 1
                elif piece_type == 'q':
                    piece_counts[f'{color}_queens'] += 1
                elif piece_type == 'k':
                    piece_counts[f'{color}_kings'] += 1
        
        total_pieces = sum(piece_counts.values())
        
        return {
            "total_pieces": total_pieces,
            "piece_counts": piece_counts,
            "legal_position": board.is_valid(),
            "check": board.is_check(),
            "checkmate": board.is_checkmate(),
            "stalemate": board.is_stalemate()
        }
        
    except Exception as e:
        return {"error": f"Position analysis failed: {e}"}

def main():
    """Run comprehensive accuracy tests."""
    print("🎯 Full API Accuracy Test")
    print("=" * 50)
    print("Testing both occupancy and piece classification accuracy")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running on port 8000")
            return
    except:
        print("❌ API is not running on port 8000")
        return
    
    print("✅ API is running")
    
    # Test each position
    results = []
    total_tests = len(TEST_POSITIONS)
    successful_tests = 0
    fen_matches = 0
    total_processing_time = 0
    
    for i, position in enumerate(TEST_POSITIONS, 1):
        print(f"\n📊 Test {i}/{total_tests}")
        result = test_api_position(position)
        results.append({
            "position": position['name'],
            "result": result
        })
        
        if result.get("success"):
            successful_tests += 1
            total_processing_time += result.get("processing_time", 0)
            if result.get("fen_match"):
                fen_matches += 1
            
            # Analyze the predicted position
            if result.get("predicted_fen"):
                analysis = analyze_chess_position(result["predicted_fen"])
                print(f"   📈 Position Analysis:")
                print(f"      Total Pieces: {analysis.get('total_pieces', 'N/A')}")
                print(f"      Legal Position: {analysis.get('legal_position', 'N/A')}")
                print(f"      In Check: {analysis.get('check', 'N/A')}")
                print(f"      Checkmate: {analysis.get('checkmate', 'N/A')}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ACCURACY SUMMARY")
    print("=" * 50)
    
    success_rate = (successful_tests / total_tests) * 100
    fen_accuracy = (fen_matches / successful_tests) * 100 if successful_tests > 0 else 0
    avg_processing_time = total_processing_time / successful_tests if successful_tests > 0 else 0
    
    print(f"✅ Successful Tests: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"🎯 FEN Accuracy: {fen_matches}/{successful_tests} ({fen_accuracy:.1f}%)")
    print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        pos_name = result["position"]
        res = result["result"]
        
        if res.get("success"):
            status = "✅" if res.get("fen_match") else "⚠️"
            print(f"   {i}. {pos_name}: {status} {res.get('processing_time', 0):.2f}s")
        else:
            print(f"   {i}. {pos_name}: ❌ {res.get('error', 'Unknown error')}")
    
    # Performance assessment
    print(f"\n🚀 PERFORMANCE ASSESSMENT:")
    if avg_processing_time < 2.0:
        print("   ⚡ Excellent speed (< 2s per position)")
    elif avg_processing_time < 5.0:
        print("   ✅ Good speed (2-5s per position)")
    else:
        print("   ⚠️  Slow processing (> 5s per position)")
    
    if fen_accuracy >= 90:
        print("   🎯 Excellent accuracy (≥90%)")
    elif fen_accuracy >= 75:
        print("   ✅ Good accuracy (75-90%)")
    elif fen_accuracy >= 50:
        print("   ⚠️  Moderate accuracy (50-75%)")
    else:
        print("   ❌ Poor accuracy (<50%)")
    
    print(f"\n🎉 Test completed! Your new piece classifier is working with {fen_accuracy:.1f}% accuracy.")

if __name__ == "__main__":
    main()
