#!/usr/bin/env python3
"""
Real-world performance evaluation using grey background test/validation images.
This gives you the accurate breakdown of each model component on unseen data.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time
import random
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorldPerformanceEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        self.metrics = {
            'total_images': 0,
            'successful_api_calls': 0,
            'total_squares': 0,
            'occupied_squares_detected': 0,
            'pieces_detected': 0,
            'high_confidence_color': 0,
            'high_confidence_piece': 0,
            'perfect_fen_images': 0,
            'ground_truth_pieces': 0,
            'correct_occupancy': 0,
            'correct_color': 0,
            'correct_piece': 0
        }
        
    def evaluate_real_world_performance(self, sample_size=10, use_test=True, use_val=True):
        """Evaluate real-world performance on test/validation images"""
        print("🌍 REAL-WORLD PERFORMANCE EVALUATION")
        print("=" * 60)
        print("Testing on grey background test/validation images")
        print("(Images NOT used for training)")
        print()
        
        # Check API health
        if not self.check_api_health():
            return False
        
        # Find test/validation images
        test_images = self.find_test_validation_images(use_test, use_val)
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
        self.generate_real_world_report()
        
        return True
    
    def check_api_health(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Status: {health_data.get('status', 'Unknown')}")
                print(f"   Models loaded: Occupancy={health_data.get('occupancy_model_loaded', False)}, "
                      f"Color={health_data.get('color_model_loaded', False)}, "
                      f"Piece={health_data.get('piece_type_model_loaded', False)}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    def find_test_validation_images(self, use_test=True, use_val=True):
        """Find test and validation images with their annotations"""
        test_images = []
        
        # Test images
        if use_test:
            test_dir = Path("grey_background_dataset/images/test")
            test_ann_dir = Path("grey_background_dataset/annotations/test")
            
            if test_dir.exists():
                for img_path in test_dir.glob("*.JPG"):
                    ann_path = test_ann_dir / f"{img_path.stem}.json"
                    if ann_path.exists():
                        test_images.append((str(img_path), str(ann_path)))
                        
                print(f"📁 Found {len([img for img, _ in test_images if 'test' in img])} test images")
        
        # Validation images
        if use_val:
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
            
            # Extract ground truth
            ground_truth = {
                'fen': annotation.get('fen', ''),
                'corners': annotation.get('corners', []),
                'pieces': annotation.get('pieces', {}),
                'occupancy': []
            }
            
            # Create occupancy array (64 squares)
            for rank in range(8, 0, -1):  # 8 to 1
                for file in range(8):  # a to h
                    square_name = f"{chr(97+file)}{rank}"
                    is_occupied = square_name in ground_truth['pieces'] and ground_truth['pieces'][square_name] is not None
                    ground_truth['occupancy'].append(is_occupied)
            
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error loading ground truth from {annotation_path}: {e}")
            return None
    
    def evaluate_single_image(self, image_path, annotation_path):
        """Evaluate single image with ground truth comparison"""
        # Load ground truth
        ground_truth = self.load_ground_truth(annotation_path)
        if ground_truth is None:
            print(f"  ❌ Could not load ground truth")
            return None
        
        # Use ground truth corners if available, otherwise detect
        corners = ground_truth.get('corners', [])
        if not corners:
            corners = self.detect_corners(image_path)
            if corners is None:
                print(f"  ❌ Could not detect corners")
                return None
        
        # Call API
        api_result = self.call_api(image_path, corners)
        if api_result is None:
            print(f"  ❌ API call failed")
            return None
        
        # Compare with ground truth
        comparison = self.compare_with_ground_truth(api_result, ground_truth)
        
        # Calculate metrics for this image
        result = {
            'image': Path(image_path).name,
            'api_result': api_result,
            'ground_truth': ground_truth,
            'comparison': comparison,
            'success': True
        }
        
        # Print results
        print(f"  🎯 Pieces detected: {api_result['pieces_detected']} (GT: {len([p for p in ground_truth['pieces'].values() if p])})")
        print(f"  📊 Occupancy accuracy: {comparison['occupancy_accuracy']:.1f}%")
        print(f"  🎨 Color accuracy: {comparison['color_accuracy']:.1f}%")
        print(f"  ♟️  Piece accuracy: {comparison['piece_accuracy']:.1f}%")
        print(f"  🎯 FEN match: {'Yes' if comparison['fen_match'] else 'No'}")
        
        return result
    
    def detect_corners(self, image_path):
        """Detect chessboard corners"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try different chessboard sizes
            for pattern_size in [(7, 7), (8, 8), (9, 9)]:
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    corners_2d = corners.reshape(-1, 2)
                    
                    top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
                    top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
                    bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
                    bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
                    
                    return [top_left.tolist(), top_right.tolist(), bottom_right.tolist(), bottom_left.tolist()]
            
            # Fallback to estimated corners
            h, w = img.shape[:2]
            margin = min(h, w) * 0.1
            return [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ]
        except:
            return None
    
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
                    'total_squares': len(occupancy),
                    'success': result.get('success', False)
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
            'correct_occupancy_count': 0,
            'correct_color_count': 0,
            'correct_piece_count': 0,
            'ground_truth_occupied_count': 0
        }
        
        # Compare occupancy (square by square)
        gt_occupancy = ground_truth['occupancy']
        api_occupancy = api_result['occupancy']
        
        if len(gt_occupancy) == len(api_occupancy):
            correct_occupancy = sum(1 for gt, api in zip(gt_occupancy, api_occupancy) if gt == api)
            comparison['occupancy_accuracy'] = (correct_occupancy / len(gt_occupancy)) * 100
            comparison['correct_occupancy_count'] = correct_occupancy
        
        # Compare pieces (only on occupied squares)
        gt_pieces = ground_truth['pieces']
        api_pieces = api_result['pieces']
        
        # Count ground truth occupied squares
        gt_occupied_count = sum(gt_occupancy)
        comparison['ground_truth_occupied_count'] = gt_occupied_count
        
        # Compare color and piece type on occupied squares
        correct_colors = 0
        correct_pieces = 0
        
        # Convert pieces array to square-based comparison
        square_names = []
        for rank in range(8, 0, -1):  # 8 to 1
            for file in range(8):  # a to h
                square_names.append(f"{chr(97+file)}{rank}")
        
        for i, (gt_occ, api_piece) in enumerate(zip(gt_occupancy, api_pieces)):
            if gt_occ:  # Only check occupied squares in ground truth
                square_name = square_names[i]
                gt_piece = gt_pieces.get(square_name)
                
                if gt_piece and api_piece:
                    # Check color
                    gt_color = 'white' if gt_piece.isupper() else 'black'
                    api_color = 'white' if api_piece.isupper() else 'black'
                    if gt_color == api_color:
                        correct_colors += 1
                    
                    # Check piece type
                    gt_piece_type = gt_piece.lower()
                    api_piece_type = api_piece.lower()
                    if gt_piece_type == api_piece_type:
                        correct_pieces += 1
        
        if gt_occupied_count > 0:
            comparison['color_accuracy'] = (correct_colors / gt_occupied_count) * 100
            comparison['piece_accuracy'] = (correct_pieces / gt_occupied_count) * 100
            comparison['correct_color_count'] = correct_colors
            comparison['correct_piece_count'] = correct_pieces
        
        # Compare FEN
        gt_fen = ground_truth.get('fen', '').split(' ')[0]  # Just the position part
        api_fen = api_result.get('fen', '').split(' ')[0]
        comparison['fen_match'] = gt_fen == api_fen
        
        return comparison
    
    def update_metrics(self, result):
        """Update overall metrics"""
        self.metrics['total_images'] += 1
        self.metrics['successful_api_calls'] += 1
        
        api_result = result['api_result']
        comparison = result['comparison']
        
        self.metrics['total_squares'] += api_result['total_squares']
        self.metrics['occupied_squares_detected'] += api_result['occupied_squares']
        self.metrics['pieces_detected'] += api_result['pieces_detected']
        self.metrics['ground_truth_pieces'] += comparison['ground_truth_occupied_count']
        
        self.metrics['correct_occupancy'] += comparison['correct_occupancy_count']
        self.metrics['correct_color'] += comparison['correct_color_count']
        self.metrics['correct_piece'] += comparison['correct_piece_count']
        
        if comparison['fen_match']:
            self.metrics['perfect_fen_images'] += 1
    
    def generate_real_world_report(self):
        """Generate comprehensive real-world performance report"""
        if self.metrics['total_images'] == 0:
            print("❌ No results to analyze")
            return
        
        print("\n" + "=" * 80)
        print("🌍 REAL-WORLD PERFORMANCE REPORT")
        print("=" * 80)
        
        # Calculate the 4 metrics you requested
        occupancy_accuracy = (self.metrics['correct_occupancy'] / self.metrics['total_squares']) * 100
        color_accuracy = (self.metrics['correct_color'] / self.metrics['ground_truth_pieces']) * 100 if self.metrics['ground_truth_pieces'] > 0 else 0
        piece_accuracy = (self.metrics['correct_piece'] / self.metrics['ground_truth_pieces']) * 100 if self.metrics['ground_truth_pieces'] > 0 else 0
        fen_accuracy = (self.metrics['perfect_fen_images'] / self.metrics['total_images']) * 100
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Images tested: {self.metrics['total_images']}")
        print(f"   Successful API calls: {self.metrics['successful_api_calls']}")
        print(f"   Total squares analyzed: {self.metrics['total_squares']}")
        print(f"   Ground truth pieces: {self.metrics['ground_truth_pieces']}")
        print(f"   API detected pieces: {self.metrics['pieces_detected']}")
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
        print(f"   Correct color predictions: {self.metrics['correct_color']}/{self.metrics['ground_truth_pieces']}")
        print(f"   Correct piece predictions: {self.metrics['correct_piece']}/{self.metrics['ground_truth_pieces']}")
        print(f"   Perfect FEN matches: {self.metrics['perfect_fen_images']}/{self.metrics['total_images']}")
        print()
        
        # Performance assessment
        print(f"🏆 PERFORMANCE ASSESSMENT:")
        print(f"   Occupancy Detection: {'EXCELLENT' if occupancy_accuracy >= 80 else 'GOOD' if occupancy_accuracy >= 60 else 'FAIR' if occupancy_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   Color Classification: {'EXCELLENT' if color_accuracy >= 80 else 'GOOD' if color_accuracy >= 60 else 'FAIR' if color_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   Piece Classification: {'EXCELLENT' if piece_accuracy >= 80 else 'GOOD' if piece_accuracy >= 60 else 'FAIR' if piece_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print(f"   FEN Generation: {'EXCELLENT' if fen_accuracy >= 80 else 'GOOD' if fen_accuracy >= 60 else 'FAIR' if fen_accuracy >= 40 else 'NEEDS IMPROVEMENT'}")
        print()
        
        # Per-image breakdown
        print(f"📋 PER-IMAGE BREAKDOWN:")
        for i, result in enumerate(self.results):
            comparison = result['comparison']
            api_result = result['api_result']
            print(f"   Image {i+1} ({result['image']}):")
            print(f"      Pieces detected: {api_result['pieces_detected']}")
            print(f"      Occupancy: {comparison['occupancy_accuracy']:.1f}%")
            print(f"      Color: {comparison['color_accuracy']:.1f}%")
            print(f"      Piece: {comparison['piece_accuracy']:.1f}%")
            print(f"      FEN: {'✅' if comparison['fen_match'] else '❌'}")
        
        # Save detailed results
        self.save_results()
        
        print(f"\n💾 Detailed results saved to: real_world_performance_results.json")
    
    def save_results(self):
        """Save detailed results"""
        # Calculate final metrics
        occupancy_accuracy = (self.metrics['correct_occupancy'] / self.metrics['total_squares']) * 100
        color_accuracy = (self.metrics['correct_color'] / self.metrics['ground_truth_pieces']) * 100 if self.metrics['ground_truth_pieces'] > 0 else 0
        piece_accuracy = (self.metrics['correct_piece'] / self.metrics['ground_truth_pieces']) * 100 if self.metrics['ground_truth_pieces'] > 0 else 0
        fen_accuracy = (self.metrics['perfect_fen_images'] / self.metrics['total_images']) * 100
        
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
        
        with open("real_world_performance_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

def main():
    """Main function"""
    print("Real-World Performance Evaluation")
    print("=" * 50)
    print("This will test your API against grey background test/validation images")
    print("(Images NOT used for training)")
    print()
    
    evaluator = RealWorldPerformanceEvaluator()
    
    # Configuration
    sample_size = input("How many images to test? (default: 10): ").strip()
    if not sample_size:
        sample_size = 10
    else:
        sample_size = int(sample_size)
    
    use_test = input("Use test images? (y/n, default: y): ").strip().lower() != 'n'
    use_val = input("Use validation images? (y/n, default: y): ").strip().lower() != 'n'
    
    success = evaluator.evaluate_real_world_performance(sample_size, use_test, use_val)
    
    if success:
        print("\n🎯 REAL-WORLD EVALUATION COMPLETED!")
        print("You now have accurate performance metrics on unseen data.")
    else:
        print("\n❌ EVALUATION FAILED!")
        print("Please check your API and dataset.")

if __name__ == "__main__":
    main()
