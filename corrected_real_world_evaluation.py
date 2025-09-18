#!/usr/bin/env python3
"""
Corrected real-world performance evaluation using proper FEN parsing.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedRealWorldEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        self.metrics = {
            'total_images': 0,
            'successful_api_calls': 0,
            'total_squares': 0,
            'ground_truth_occupied': 0,
            'api_detected_occupied': 0,
            'correct_occupancy': 0,
            'correct_color': 0,
            'correct_piece': 0,
            'perfect_fen_images': 0
        }
        
    def parse_fen_to_board(self, fen):
        """Parse FEN notation to get board state"""
        board_state = {}
        position_part = fen.split(' ')[0]  # Get just the position part
        
        ranks = position_part.split('/')
        
        for rank_idx, rank in enumerate(ranks):
            rank_number = 8 - rank_idx  # Rank 8 is first in FEN
            file_idx = 0
            
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    square_name = f"{chr(97 + file_idx)}{rank_number}"
                    board_state[square_name] = char
                    file_idx += 1
        
        return board_state
    
    def board_state_to_arrays(self, board_state):
        """Convert board state to occupancy and pieces arrays"""
        occupancy = []
        pieces = []
        
        # Create arrays in the same order as API (a1-h1, a2-h2, ..., a8-h8)
        for rank in range(1, 9):
            for file in range(8):
                square_name = f"{chr(97 + file)}{rank}"
                
                if square_name in board_state:
                    occupancy.append(True)
                    pieces.append(board_state[square_name])
                else:
                    occupancy.append(False)
                    pieces.append(None)
        
        return occupancy, pieces
    
    def evaluate_real_world_performance(self, sample_size=10):
        """Evaluate real-world performance on test/validation images"""
        print("🌍 CORRECTED REAL-WORLD PERFORMANCE EVALUATION")
        print("=" * 60)
        print("Testing on grey background test/validation images")
        print("(Images NOT used for training)")
        print()
        
        # Check API health
        if not self.check_api_health():
            return False
        
        # Find test/validation images
        test_images = self.find_test_validation_images()
        if not test_images:
            print("❌ No test/validation images found")
            return False
        
        # Random sampling
        if len(test_images) > sample_size:
            test_images = random.sample(test_images, sample_size)
            print(f"📊 Randomly sampled {sample_size} images from {len(test_images)} available")
        
        print(f"🧪 Testing with {len(test_images)} images")
        
        # Evaluate each image
        for i, (image_path, annotation_path) in enumerate(test_images):
            print(f"\n--- Test {i+1}/{len(test_images)}: {Path(image_path).name} ---")
            result = self.evaluate_single_image(image_path, annotation_path)
            if result:
                self.results.append(result)
                self.update_metrics(result)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        return True
    
    def check_api_health(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Status: {health_data.get('status', 'Unknown')}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    def find_test_validation_images(self):
        """Find test and validation images with their annotations"""
        test_images = []
        
        # Test images
        test_dir = Path("grey_background_dataset/images/test")
        test_ann_dir = Path("grey_background_dataset/annotations/test")
        
        if test_dir.exists():
            for img_path in test_dir.glob("*.JPG"):
                ann_path = test_ann_dir / f"{img_path.stem}.json"
                if ann_path.exists():
                    test_images.append((str(img_path), str(ann_path)))
            
            print(f"📁 Found {len([img for img, _ in test_images if 'test' in img])} test images")
        
        # Validation images
        val_dir = Path("grey_background_dataset/images/val")
        val_ann_dir = Path("grey_background_dataset/annotations/val")
        
        if val_dir.exists():
            val_count = 0
            for img_path in val_dir.glob("*.JPG"):
                ann_path = val_ann_dir / f"{img_path.stem}.json"
                if ann_path.exists():
                    test_images.append((str(img_path), str(ann_path)))
                    val_count += 1
            
            print(f"📁 Found {val_count} validation images")
        
        print(f"📊 Total test/validation images: {len(test_images)}")
        return test_images
    
    def load_ground_truth(self, annotation_path):
        """Load ground truth from annotation file"""
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            # Parse FEN to get board state
            fen = annotation.get('fen', '')
            corners = annotation.get('corners', [])
            
            if fen:
                board_state = self.parse_fen_to_board(fen)
                gt_occupancy, gt_pieces = self.board_state_to_arrays(board_state)
                
                return {
                    'fen': fen,
                    'corners': corners,
                    'board_state': board_state,
                    'occupancy': gt_occupancy,
                    'pieces': gt_pieces,
                    'piece_count': sum(gt_occupancy)
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading ground truth from {annotation_path}: {e}")
            return None
    
    def evaluate_single_image(self, image_path, annotation_path):
        """Evaluate single image with proper ground truth comparison"""
        # Load ground truth
        ground_truth = self.load_ground_truth(annotation_path)
        if ground_truth is None:
            print(f"  ❌ Could not load ground truth")
            return None
        
        # Use ground truth corners
        corners = ground_truth['corners']
        
        # Call API
        api_result = self.call_api(image_path, corners)
        if api_result is None:
            print(f"  ❌ API call failed")
            return None
        
        # Compare with ground truth
        comparison = self.compare_with_ground_truth(api_result, ground_truth)
        
        # Print results
        print(f"  🎯 Pieces detected: {api_result['pieces_detected']} (GT: {ground_truth['piece_count']})")
        print(f"  📊 Occupancy accuracy: {comparison['occupancy_accuracy']:.1f}%")
        print(f"  🎨 Color accuracy: {comparison['color_accuracy']:.1f}%")
        print(f"  ♟️  Piece accuracy: {comparison['piece_accuracy']:.1f}%")
        print(f"  🎯 FEN match: {'Yes' if comparison['fen_match'] else 'No'}")
        
        return {
            'image': Path(image_path).name,
            'api_result': api_result,
            'ground_truth': ground_truth,
            'comparison': comparison
        }
    
    def call_api(self, image_path, corners):
        """Call API and return results"""
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'corners': json.dumps(corners),
                    'debug': 'true'
                }
                
                response = requests.post(
                    f"{self.api_url}/recognize_chess_position_with_corners",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                
                pieces = result.get('pieces', [])
                occupancy = result.get('occupancy', [])
                fen = result.get('fen', '')
                
                return {
                    'pieces': pieces,
                    'occupancy': occupancy,
                    'fen': fen,
                    'pieces_detected': sum(1 for p in pieces if p is not None),
                    'occupied_squares': sum(occupancy),
                    'total_squares': len(occupancy)
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"API call error: {e}")
            return None
    
    def compare_with_ground_truth(self, api_result, ground_truth):
        """Compare API results with ground truth"""
        comparison = {
            'occupancy_accuracy': 0,
            'color_accuracy': 0,
            'piece_accuracy': 0,
            'fen_match': False,
            'correct_occupancy': 0,
            'correct_color': 0,
            'correct_piece': 0
        }
        
        # Compare occupancy
        gt_occupancy = ground_truth['occupancy']
        api_occupancy = api_result['occupancy']
        
        if len(gt_occupancy) == len(api_occupancy):
            correct_occupancy = sum(1 for gt, api in zip(gt_occupancy, api_occupancy) if gt == api)
            comparison['occupancy_accuracy'] = (correct_occupancy / len(gt_occupancy)) * 100
            comparison['correct_occupancy'] = correct_occupancy
        
        # Compare pieces on occupied squares (ground truth)
        gt_pieces = ground_truth['pieces']
        api_pieces = api_result['pieces']
        gt_occupied_count = sum(gt_occupancy)
        
        correct_colors = 0
        correct_pieces = 0
        
        if len(gt_pieces) == len(api_pieces):
            for i, (gt_occ, gt_piece, api_piece) in enumerate(zip(gt_occupancy, gt_pieces, api_pieces)):
                if gt_occ and gt_piece:  # Only check squares that should have pieces
                    if api_piece:  # API detected a piece
                        # Check color
                        gt_color = 'white' if gt_piece.isupper() else 'black'
                        api_color = 'white' if api_piece.isupper() else 'black'
                        if gt_color == api_color:
                            correct_colors += 1
                        
                        # Check piece type
                        if gt_piece.lower() == api_piece.lower():
                            correct_pieces += 1
        
        if gt_occupied_count > 0:
            comparison['color_accuracy'] = (correct_colors / gt_occupied_count) * 100
            comparison['piece_accuracy'] = (correct_pieces / gt_occupied_count) * 100
            comparison['correct_color'] = correct_colors
            comparison['correct_piece'] = correct_pieces
        
        # Compare FEN (position part only)
        gt_fen_pos = ground_truth['fen'].split(' ')[0]
        api_fen_pos = api_result['fen'].split(' ')[0]
        comparison['fen_match'] = gt_fen_pos == api_fen_pos
        
        return comparison
    
    def update_metrics(self, result):
        """Update overall metrics"""
        self.metrics['total_images'] += 1
        self.metrics['successful_api_calls'] += 1
        
        api_result = result['api_result']
        ground_truth = result['ground_truth']
        comparison = result['comparison']
        
        self.metrics['total_squares'] += api_result['total_squares']
        self.metrics['ground_truth_occupied'] += ground_truth['piece_count']
        self.metrics['api_detected_occupied'] += api_result['occupied_squares']
        
        self.metrics['correct_occupancy'] += comparison['correct_occupancy']
        self.metrics['correct_color'] += comparison['correct_color']
        self.metrics['correct_piece'] += comparison['correct_piece']
        
        if comparison['fen_match']:
            self.metrics['perfect_fen_images'] += 1
    
    def generate_comprehensive_report(self):
        """Generate comprehensive real-world performance report"""
        if self.metrics['total_images'] == 0:
            print("❌ No results to analyze")
            return
        
        print("\n" + "=" * 80)
        print("🌍 CORRECTED REAL-WORLD PERFORMANCE REPORT")
        print("=" * 80)
        
        # Calculate the 4 metrics you requested
        occupancy_accuracy = (self.metrics['correct_occupancy'] / self.metrics['total_squares']) * 100
        color_accuracy = (self.metrics['correct_color'] / self.metrics['ground_truth_occupied']) * 100 if self.metrics['ground_truth_occupied'] > 0 else 0
        piece_accuracy = (self.metrics['correct_piece'] / self.metrics['ground_truth_occupied']) * 100 if self.metrics['ground_truth_occupied'] > 0 else 0
        fen_accuracy = (self.metrics['perfect_fen_images'] / self.metrics['total_images']) * 100
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Images tested: {self.metrics['total_images']}")
        print(f"   Total squares analyzed: {self.metrics['total_squares']}")
        print(f"   Ground truth occupied squares: {self.metrics['ground_truth_occupied']}")
        print(f"   API detected occupied squares: {self.metrics['api_detected_occupied']}")
        print()
        
        print(f"🎯 YOUR REQUESTED 4 METRICS:")
        print(f"   1. % of squares where occupancy is correct: {occupancy_accuracy:.1f}%")
        print(f"   2. % of occupied squares where color is correct: {color_accuracy:.1f}%")
        print(f"   3. % of occupied squares where piece is correct: {piece_accuracy:.1f}%")
        print(f"   4. % of images where entire FEN is 100% correct: {fen_accuracy:.1f}%")
        print()
        
        # Detailed breakdown
        print(f"🔧 DETAILED BREAKDOWN:")
        print(f"   Correct occupancy predictions: {self.metrics['correct_occupancy']}/{self.metrics['total_squares']}")
        print(f"   Correct color predictions: {self.metrics['correct_color']}/{self.metrics['ground_truth_occupied']}")
        print(f"   Correct piece predictions: {self.metrics['correct_piece']}/{self.metrics['ground_truth_occupied']}")
        print(f"   Perfect FEN matches: {self.metrics['perfect_fen_images']}/{self.metrics['total_images']}")
        print()
        
        # Performance assessment
        print(f"🏆 PERFORMANCE ASSESSMENT:")
        print(f"   Occupancy Detection: {'EXCELLENT' if occupancy_accuracy >= 80 else 'GOOD' if occupancy_accuracy >= 60 else 'FAIR' if occupancy_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   Color Classification: {'EXCELLENT' if color_accuracy >= 80 else 'GOOD' if color_accuracy >= 60 else 'FAIR' if color_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   Piece Classification: {'EXCELLENT' if piece_accuracy >= 80 else 'GOOD' if piece_accuracy >= 60 else 'FAIR' if piece_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   FEN Generation: {'EXCELLENT' if fen_accuracy >= 80 else 'GOOD' if fen_accuracy >= 60 else 'FAIR' if fen_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print()
        
        # Model component analysis
        print(f"🔍 MODEL COMPONENT ANALYSIS:")
        
        # Occupancy analysis
        occupancy_precision = (self.metrics['correct_occupancy'] / self.metrics['api_detected_occupied']) * 100 if self.metrics['api_detected_occupied'] > 0 else 0
        occupancy_recall = (self.metrics['correct_occupancy'] / self.metrics['ground_truth_occupied']) * 100 if self.metrics['ground_truth_occupied'] > 0 else 0
        
        print(f"   Occupancy Model:")
        print(f"     - Accuracy: {occupancy_accuracy:.1f}% (correct predictions)")
        print(f"     - Precision: {occupancy_precision:.1f}% (of detected, how many correct)")
        print(f"     - Recall: {occupancy_recall:.1f}% (of actual pieces, how many found)")
        
        print(f"   Color Model:")
        print(f"     - Accuracy: {color_accuracy:.1f}% (when pieces are present)")
        
        print(f"   Piece Model:")
        print(f"     - Accuracy: {piece_accuracy:.1f}% (when pieces are present)")
        
        # Per-image breakdown
        print(f"\n📋 PER-IMAGE BREAKDOWN:")
        for i, result in enumerate(self.results):
            comparison = result['comparison']
            api_result = result['api_result']
            ground_truth = result['ground_truth']
            
            print(f"   Image {i+1} ({result['image']}):")
            print(f"      GT pieces: {ground_truth['piece_count']}, API detected: {api_result['pieces_detected']}")
            print(f"      Occupancy: {comparison['occupancy_accuracy']:.1f}%")
            print(f"      Color: {comparison['color_accuracy']:.1f}%")
            print(f"      Piece: {comparison['piece_accuracy']:.1f}%")
            print(f"      FEN: {'✅' if comparison['fen_match'] else '❌'}")
            print(f"      GT FEN: {ground_truth['fen'][:50]}...")
            print(f"      API FEN: {api_result['fen'][:50]}...")
        
        # Save detailed results
        self.save_results(occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy)
        
        print(f"\n💾 Detailed results saved to: corrected_real_world_results.json")
    
    def save_results(self, occupancy_accuracy, color_accuracy, piece_accuracy, fen_accuracy):
        """Save detailed results"""
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_url': self.api_url,
            'dataset': 'grey_background_dataset (test/val)',
            'summary_metrics': {
                'occupancy_accuracy': occupancy_accuracy,
                'color_accuracy': color_accuracy,
                'piece_accuracy': piece_accuracy,
                'fen_accuracy': fen_accuracy
            },
            'raw_metrics': self.metrics,
            'detailed_results': self.results
        }
        
        with open("corrected_real_world_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

def main():
    """Main function"""
    print("Corrected Real-World Performance Evaluation")
    print("=" * 60)
    print("This will test your API against grey background test/validation images")
    print("with proper FEN parsing for accurate ground truth comparison.")
    print()
    
    evaluator = CorrectedRealWorldEvaluator()
    
    sample_size = input("How many images to test? (default: 10): ").strip()
    if not sample_size:
        sample_size = 10
    else:
        sample_size = int(sample_size)
    
    success = evaluator.evaluate_real_world_performance(sample_size)
    
    if success:
        print("\n🎯 REAL-WORLD EVALUATION COMPLETED!")
        print("You now have accurate performance metrics on unseen test/validation data.")
    else:
        print("\n❌ EVALUATION FAILED!")
        print("Please check your API and dataset.")

if __name__ == "__main__":
    main()
