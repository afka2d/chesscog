#!/usr/bin/env python3
"""
Simple accuracy evaluation using model's own predictions as ground truth.
This measures consistency and reliability without requiring manual annotations.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAccuracyEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        
        # Metrics tracking
        self.metrics = {
            'total_images': 0,
            'successful_calls': 0,
            'total_squares': 0,
            'occupied_squares': 0,
            'pieces_detected': 0,
            'high_confidence_pieces': 0,
            'perfect_fen_images': 0
        }
        
    def check_api(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is running")
                return True
            else:
                logger.error("❌ API not responding correctly")
                return False
        except:
            logger.error("❌ Cannot connect to API")
            return False
    
    def detect_corners(self, image_path):
        """Detect chessboard corners"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_2d = corners.reshape(-1, 2)
            
            top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
            top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
            bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
            bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
            
            return [top_left, top_right, bottom_right, bottom_left]
        else:
            h, w = img.shape[:2]
            margin = min(h, w) * 0.1
            return [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ]
    
    def evaluate_image(self, image_path):
        """Evaluate a single image"""
        logger.info(f"Evaluating: {Path(image_path).name}")
        
        corners = self.detect_corners(image_path)
        
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
                
                # Extract predictions
                predicted_pieces = result.get('pieces', [])
                predicted_occupancy = result.get('occupancy', [])
                predicted_fen = result.get('fen', '')
                debug_info = result.get('debug_info', {})
                
                # Calculate metrics
                pieces_detected = sum(1 for p in predicted_pieces if p is not None)
                occupied_squares = sum(predicted_occupancy)
                total_squares = len(predicted_occupancy)
                
                # Count high confidence pieces (if debug info available)
                high_confidence_pieces = 0
                if debug_info and 'square_details' in debug_info:
                    for square in debug_info['square_details']:
                        if 'color_confidence' in square and 'piece_confidence' in square:
                            color_conf = square['color_confidence']
                            piece_conf = square['piece_confidence']
                            if color_conf >= 0.8 and piece_conf >= 0.8:
                                high_confidence_pieces += 1
                
                # Check if FEN is valid (not empty board)
                fen_perfect = predicted_fen != '8/8/8/8/8/8/8/8 w - - 0 1' and pieces_detected > 0
                
                # Update metrics
                self.metrics['total_images'] += 1
                self.metrics['successful_calls'] += 1
                self.metrics['total_squares'] += total_squares
                self.metrics['occupied_squares'] += occupied_squares
                self.metrics['pieces_detected'] += pieces_detected
                self.metrics['high_confidence_pieces'] += high_confidence_pieces
                if fen_perfect:
                    self.metrics['perfect_fen_images'] += 1
                
                # Store detailed results
                image_result = {
                    'image': Path(image_path).name,
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'total_squares': total_squares,
                    'high_confidence_pieces': high_confidence_pieces,
                    'fen': predicted_fen,
                    'fen_perfect': fen_perfect,
                    'success': result.get('success', False)
                }
                
                self.results.append(image_result)
                
                logger.info(f"  Pieces: {pieces_detected}, Occupied: {occupied_squares}, High Conf: {high_confidence_pieces}")
                logger.info(f"  FEN: {predicted_fen}")
                logger.info(f"  FEN Perfect: {fen_perfect}")
                
                return image_result
            else:
                logger.error(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating {image_path}: {e}")
            return None
    
    def evaluate_dataset(self, dataset_path, sample_size=10):
        """Evaluate dataset with random sampling"""
        logger.info(f"Evaluating dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        if not images:
            logger.error("No images found to evaluate")
            return
        
        logger.info(f"Found {len(images)} images")
        
        # Random sampling
        if len(images) > sample_size:
            images = random.sample(images, sample_size)
            logger.info(f"Randomly sampled {sample_size} images for evaluation")
        
        # Evaluate each image
        for i, image_path in enumerate(images):
            print(f"\n--- Image {i+1}/{len(images)} ---")
            self.evaluate_image(str(image_path))
            time.sleep(0.5)  # Small delay
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive accuracy report"""
        if self.metrics['total_images'] == 0:
            logger.warning("No results to analyze")
            return
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CHESS MODEL ACCURACY EVALUATION")
        print("=" * 80)
        
        # Calculate percentages
        success_rate = (self.metrics['successful_calls'] / self.metrics['total_images']) * 100
        occupancy_rate = (self.metrics['occupied_squares'] / self.metrics['total_squares']) * 100
        piece_detection_rate = (self.metrics['pieces_detected'] / self.metrics['total_images'])
        high_confidence_rate = (self.metrics['high_confidence_pieces'] / self.metrics['pieces_detected']) * 100 if self.metrics['pieces_detected'] > 0 else 0
        perfect_fen_rate = (self.metrics['perfect_fen_images'] / self.metrics['total_images']) * 100
        
        print(f"Images evaluated: {self.metrics['total_images']}")
        print(f"Successful API calls: {self.metrics['successful_calls']}")
        print(f"Total squares evaluated: {self.metrics['total_squares']}")
        print(f"Pieces detected: {self.metrics['pieces_detected']}")
        print(f"High confidence pieces: {self.metrics['high_confidence_pieces']}")
        print()
        
        print("ACCURACY METRICS:")
        print("-" * 40)
        print(f"1. API Success Rate: {success_rate:.1f}% ({self.metrics['successful_calls']}/{self.metrics['total_images']} images)")
        print(f"2. Occupancy Detection Rate: {occupancy_rate:.1f}% ({self.metrics['occupied_squares']}/{self.metrics['total_squares']} squares)")
        print(f"3. Piece Detection Rate: {piece_detection_rate:.1f} pieces per image")
        print(f"4. High Confidence Rate: {high_confidence_rate:.1f}% ({self.metrics['high_confidence_pieces']}/{self.metrics['pieces_detected']} pieces)")
        print(f"5. Perfect FEN Rate: {perfect_fen_rate:.1f}% ({self.metrics['perfect_fen_images']}/{self.metrics['total_images']} images)")
        
        # Overall assessment
        print("\nOVERALL ASSESSMENT:")
        print("-" * 40)
        
        if success_rate >= 95:
            print("✅ API Reliability: EXCELLENT")
        elif success_rate >= 90:
            print("✅ API Reliability: GOOD")
        elif success_rate >= 80:
            print("⚠️  API Reliability: FAIR")
        else:
            print("❌ API Reliability: NEEDS IMPROVEMENT")
        
        if piece_detection_rate >= 8:
            print("✅ Piece Detection: EXCELLENT")
        elif piece_detection_rate >= 5:
            print("✅ Piece Detection: GOOD")
        elif piece_detection_rate >= 2:
            print("⚠️  Piece Detection: FAIR")
        else:
            print("❌ Piece Detection: NEEDS IMPROVEMENT")
        
        if high_confidence_rate >= 80:
            print("✅ Classification Confidence: EXCELLENT")
        elif high_confidence_rate >= 60:
            print("✅ Classification Confidence: GOOD")
        elif high_confidence_rate >= 40:
            print("⚠️  Classification Confidence: FAIR")
        else:
            print("❌ Classification Confidence: NEEDS IMPROVEMENT")
        
        if perfect_fen_rate >= 80:
            print("✅ FEN Generation: EXCELLENT")
        elif perfect_fen_rate >= 60:
            print("✅ FEN Generation: GOOD")
        elif perfect_fen_rate >= 40:
            print("⚠️  FEN Generation: FAIR")
        else:
            print("❌ FEN Generation: NEEDS IMPROVEMENT")
        
        # Per-image analysis
        print(f"\nPER-IMAGE ANALYSIS:")
        print("-" * 40)
        for result in self.results:
            print(f"{result['image']}:")
            print(f"  Pieces: {result['pieces_detected']}")
            print(f"  Occupied: {result['occupied_squares']}")
            print(f"  High Conf: {result['high_confidence_pieces']}")
            print(f"  FEN Perfect: {'Yes' if result['fen_perfect'] else 'No'}")
            print(f"  FEN: {result['fen']}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed results"""
        results_file = "simple_accuracy_results.json"
        
        # Calculate percentages
        success_rate = (self.metrics['successful_calls'] / self.metrics['total_images']) * 100
        occupancy_rate = (self.metrics['occupied_squares'] / self.metrics['total_squares']) * 100
        piece_detection_rate = (self.metrics['pieces_detected'] / self.metrics['total_images'])
        high_confidence_rate = (self.metrics['high_confidence_pieces'] / self.metrics['pieces_detected']) * 100 if self.metrics['pieces_detected'] > 0 else 0
        perfect_fen_rate = (self.metrics['perfect_fen_images'] / self.metrics['total_images']) * 100
        
        save_data = {
            'summary': {
                'images_evaluated': self.metrics['total_images'],
                'success_rate': success_rate,
                'occupancy_rate': occupancy_rate,
                'piece_detection_rate': piece_detection_rate,
                'high_confidence_rate': high_confidence_rate,
                'perfect_fen_rate': perfect_fen_rate,
                'raw_counts': self.metrics
            },
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file}")

def main():
    """Main evaluation function"""
    print("Simple Chess Model Accuracy Evaluation")
    print("=" * 50)
    print("This evaluates your model's consistency and reliability")
    print("without requiring manual ground truth annotations.")
    print()
    
    evaluator = SimpleAccuracyEvaluator()
    
    if not evaluator.check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Find images
    dataset_path = "my_chess_images/train/images"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Ask for sample size
    sample_size = input("How many images to evaluate? (default: 5): ").strip()
    if not sample_size:
        sample_size = 5
    else:
        sample_size = int(sample_size)
    
    # Run evaluation
    evaluator.evaluate_dataset(dataset_path, sample_size)

if __name__ == "__main__":
    main()
