#!/usr/bin/env python3
"""
Debug the piece classification issue.
"""

import requests
import json
from PIL import Image
import io
import chess
import numpy as np

def debug_piece_classification():
    """Debug piece classification with detailed output."""
    print("🔍 Debugging Piece Classification")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Make API request
    files = {'image': ('test.jpg', image_data, 'image/jpeg')}
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    try:
        print("📡 Making API request...")
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"   FEN: {result['fen']}")
            print(f"   Success: {result['success']}")
            
            if 'statistics' in result:
                stats = result['statistics']
                print(f"   Occupied squares: {stats['occupied_squares']}")
                print(f"   Unique piece types: {stats['unique_piece_types']}")
                print(f"   Diversity score: {stats['diversity_score']}")
                print(f"   Estimated accuracy: {stats['estimated_accuracy']}")
            
            # Show piece classification results
            pieces = result['pieces']
            occupied_pieces = [p for p in pieces if p is not None]
            
            print(f"\n🎯 PIECE CLASSIFICATION:")
            print(f"   Total pieces detected: {len(occupied_pieces)}")
            
            if occupied_pieces:
                piece_counts = {}
                for piece in occupied_pieces:
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1
                
                print(f"   Piece breakdown:")
                for piece, count in sorted(piece_counts.items()):
                    print(f"     {piece}: {count}")
            else:
                print("   No pieces detected")
            
            # Show board representation
            print(f"\n🏁 BOARD REPRESENTATION:")
            board = chess.Board(result['fen'])
            print(board)
            
            # Check if occupancy detection is working
            occupancy = result['occupancy']
            occupied_count = sum(occupancy)
            print(f"\n🔍 OCCUPANCY ANALYSIS:")
            print(f"   Total occupied squares: {occupied_count}/64")
            print(f"   Occupancy rate: {occupied_count/64*100:.1f}%")
            
            # Show which squares are occupied
            print(f"\n📍 OCCUPIED SQUARES:")
            for i, is_occupied in enumerate(occupancy):
                if is_occupied:
                    rank = 8 - (i // 8)
                    file = chr(ord('a') + (i % 8))
                    square_name = f"{file}{rank}"
                    piece = pieces[i]
                    print(f"   {square_name}: {piece}")
            
            # Check for knight bias
            if occupied_pieces:
                knight_count = sum(1 for p in occupied_pieces if 'n' in p.lower())
                total_pieces = len(occupied_pieces)
                knight_ratio = knight_count / total_pieces
                print(f"\n⚠️  KNIGHT BIAS ANALYSIS:")
                print(f"   Knights detected: {knight_count}/{total_pieces} ({knight_ratio*100:.1f}%)")
                if knight_ratio > 0.7:
                    print("   🚨 HIGH KNIGHT BIAS DETECTED - Model may be overfitting to knights")
                elif knight_ratio > 0.5:
                    print("   ⚠️  Moderate knight bias detected")
                else:
                    print("   ✅ Knight distribution looks reasonable")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_piece_classification()
