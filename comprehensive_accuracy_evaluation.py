#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation for chess model.
Measures the 4 specific metrics requested:
1. % of squares where occupancy is correct
2. % of occupied squares where color is correct  
3. % of occupied squares where piece is correct
4. % of images where entire FEN is 100% correct
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

class ComprehensiveAccuracyEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        
        # Metrics tracking
        self.metrics = {
            'occupancy_correct': 0,
            'occupancy_total': 0,
            'color_correct': 0,
            'color_total': 0,
            'piece_correct': 0,
            'piece_total': 0,
            'fen_perfect': 0,
            'images_total': 0
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
    
    def load_ground_truth(self, image_path):
        """Load ground truth annotation for an image"""
        # Look for annotation file
        annotation_path = image_path.replace('.JPG', '.json').replace('.jpg', '.json')
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                return json.load(f)
        
        # If no annotation file, return None (will skip this image)
        return None
    
    def generate_ground_truth_fen(self, ground_truth):
        """Generate FEN from ground truth annotations"""
        if not ground_truth:
            return None
        
        # Create board representation
        board = [['' for _ in range(8)] for _ in range(8)]
        
        for square_name, annotation in ground_truth.items():
            if annotation['occupied']:
                file = ord(square_name[0]) - ord('a')
                rank = 8 - int(square_name[1])
                
                if 0 <= file < 8 and 0 <= rank < 8:
                    piece_char = self.get_piece_char(annotation['color'], annotation['piece'])
                    board[rank][file] = piece_char
        
        # Convert to FEN format
        fen_parts = []
        for rank in board:
            fen_rank = ""
            empty_count = 0
            
            for square in rank:
                if square == '':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += square
            
            if empty_count > 0:
                fen_rank += str(empty_count)
            
            fen_parts.append(fen_rank)
        
        return '/'.join(fen_parts) + ' w - - 0 1'
    
    def get_piece_char(self, color, piece):
        """Convert color and piece to FEN character"""
        piece_map = {
            ('white', 'pawn'): 'P',
            ('white', 'rook'): 'R',
            ('white', 'knight'): 'N',
            ('white', 'bishop'): 'B',
            ('white', 'queen'): 'Q',
            ('white', 'king'): 'K',
            ('black', 'pawn'): 'p',
            ('black', 'rook'): 'r',
            ('black', 'knight'): 'n',
            ('black', 'bishop'): 'b',
            ('black', 'queen'): 'q',
            ('black', 'king'): 'k'
        }
        
        return piece_map.get((color, piece), '?')
    
    def evaluate_image(self, image_path, ground_truth):
        """Evaluate a single image against ground truth"""
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
                
                # Generate ground truth FEN
                gt_fen = self.generate_ground_truth_fen(ground_truth)
                
                # Evaluate each square
                image_metrics = {
                    'occupancy_correct': 0,
                    'occupancy_total': 0,
                    'color_correct': 0,
                    'color_total': 0,
                    'piece_correct': 0,
                    'piece_total': 0,
                    'fen_perfect': 0
                }
                
                # Process each square
                for rank in range(8):
                    for file in range(8):
                        square_name = f"{chr(97+file)}{8-rank}"
                        square_idx = rank * 8 + file
                        
                        # Get ground truth for this square
                        gt_square = ground_truth.get(square_name, {})
                        gt_occupied = gt_square.get('occupied', False)
                        gt_color = gt_square.get('color', None)
                        gt_piece = gt_square.get('piece', None)
                        
                        # Get predictions for this square
                        pred_occupied = predicted_occupancy[square_idx] == 1 if square_idx < len(predicted_occupancy) else False
                        pred_piece = predicted_pieces[square_idx] if square_idx < len(predicted_pieces) else None
                        
                        # Parse predicted piece
                        pred_color = None
                        pred_piece_type = None
                        if pred_piece:
                            # Parse piece string like "White Rook" or "Black Pawn"
                            parts = pred_piece.split()
                            if len(parts) == 2:
                                pred_color = parts[0].lower()
                                pred_piece_type = parts[1].lower()
                        
                        # Evaluate occupancy
                        image_metrics['occupancy_total'] += 1
                        if pred_occupied == gt_occupied:
                            image_metrics['occupancy_correct'] += 1
                        
                        # Evaluate color and piece for occupied squares
                        if gt_occupied and pred_occupied:
                            # Color evaluation
                            if gt_color and pred_color:
                                image_metrics['color_total'] += 1
                                if pred_color == gt_color:
                                    image_metrics['color_correct'] += 1
                            
                            # Piece evaluation
                            if gt_piece and pred_piece_type:
                                image_metrics['piece_total'] += 1
                                if pred_piece_type == gt_piece:
                                    image_metrics['piece_correct'] += 1
                
                # Evaluate FEN accuracy
                if gt_fen and predicted_fen:
                    # Normalize FENs for comparison (remove move counts, etc.)
                    gt_fen_normalized = gt_fen.split()[0]  # Just the board part
                    pred_fen_normalized = predicted_fen.split()[0]  # Just the board part
                    
                    if gt_fen_normalized == pred_fen_normalized:
                        image_metrics['fen_perfect'] = 1
                
                # Update global metrics
                for key in image_metrics:
                    self.metrics[key] += image_metrics[key]
                
                self.metrics['images_total'] += 1
                
                # Store detailed results
                image_result = {
                    'image': Path(image_path).name,
                    'metrics': image_metrics,
                    'predicted_fen': predicted_fen,
                    'ground_truth_fen': gt_fen,
                    'fen_match': image_metrics['fen_perfect'] == 1
                }
                
                self.results.append(image_result)
                
                logger.info(f"  Occupancy: {image_metrics['occupancy_correct']}/{image_metrics['occupancy_total']}")
                logger.info(f"  Color: {image_metrics['color_correct']}/{image_metrics['color_total']}")
                logger.info(f"  Piece: {image_metrics['piece_correct']}/{image_metrics['piece_total']}")
                logger.info(f"  FEN Perfect: {image_metrics['fen_perfect']}")
                
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
        
        # Find all images with ground truth
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        # Filter images that have ground truth annotations
        images_with_gt = []
        for image_path in images:
            gt = self.load_ground_truth(str(image_path))
            if gt:
                images_with_gt.append(str(image_path))
        
        if not images_with_gt:
            logger.error("No images with ground truth annotations found")
            return
        
        logger.info(f"Found {len(images_with_gt)} images with ground truth")
        
        # Random sampling
        if len(images_with_gt) > sample_size:
            images_with_gt = random.sample(images_with_gt, sample_size)
            logger.info(f"Randomly sampled {sample_size} images for evaluation")
        
        # Evaluate each image
        for i, image_path in enumerate(images_with_gt):
            print(f"\n--- Image {i+1}/{len(images_with_gt)} ---")
            
            ground_truth = self.load_ground_truth(image_path)
            self.evaluate_image(image_path, ground_truth)
            
            time.sleep(0.5)  # Small delay
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive accuracy report"""
        if self.metrics['images_total'] == 0:
            logger.warning("No results to analyze")
            return
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CHESS MODEL ACCURACY EVALUATION")
        print("=" * 80)
        
        # Calculate percentages
        occupancy_accuracy = (self.metrics['occupancy_correct'] / self.metrics['occupancy_total']) * 100
        color_accuracy = (self.metrics['color_correct'] / self.metrics['color_total']) * 100 if self.metrics['color_total'] > 0 else 0
        piece_accuracy = (self.metrics['piece_correct'] / self.metrics['piece_total']) * 100 if self.metrics['piece_total'] > 0 else 0
        fen_accuracy = (self.metrics['fen_perfect'] / self.metrics['images_total']) * 100
        
        print(f"Images evaluated: {self.metrics['images_total']}")
        print(f"Total squares evaluated: {self.metrics['occupancy_total']}")
        print(f"Occupied squares evaluated: {self.metrics['color_total']} (color), {self.metrics['piece_total']} (piece)")
        print()
        
        print("ACCURACY METRICS:")
        print("-" * 40)
        print(f"1. Occupancy Accuracy: {occupancy_accuracy:.1f}% ({self.metrics['occupancy_correct']}/{self.metrics['occupancy_total']} squares)")
        print(f"2. Color Accuracy: {color_accuracy:.1f}% ({self.metrics['color_correct']}/{self.metrics['color_total']} occupied squares)")
        print(f"3. Piece Accuracy: {piece_accuracy:.1f}% ({self.metrics['piece_correct']}/{self.metrics['piece_total']} occupied squares)")
        print(f"4. Perfect FEN Accuracy: {fen_accuracy:.1f}% ({self.metrics['fen_perfect']}/{self.metrics['images_total']} images)")
        
        # Overall assessment
        print("\nOVERALL ASSESSMENT:")
        print("-" * 40)
        
        if occupancy_accuracy >= 90:
            print("✅ Occupancy detection: EXCELLENT")
        elif occupancy_accuracy >= 80:
            print("✅ Occupancy detection: GOOD")
        elif occupancy_accuracy >= 70:
            print("⚠️  Occupancy detection: FAIR")
        else:
            print("❌ Occupancy detection: NEEDS IMPROVEMENT")
        
        if color_accuracy >= 90:
            print("✅ Color classification: EXCELLENT")
        elif color_accuracy >= 80:
            print("✅ Color classification: GOOD")
        elif color_accuracy >= 70:
            print("⚠️  Color classification: FAIR")
        else:
            print("❌ Color classification: NEEDS IMPROVEMENT")
        
        if piece_accuracy >= 90:
            print("✅ Piece classification: EXCELLENT")
        elif piece_accuracy >= 80:
            print("✅ Piece classification: GOOD")
        elif piece_accuracy >= 70:
            print("⚠️  Piece classification: FAIR")
        else:
            print("❌ Piece classification: NEEDS IMPROVEMENT")
        
        if fen_accuracy >= 80:
            print("✅ FEN generation: EXCELLENT")
        elif fen_accuracy >= 60:
            print("✅ FEN generation: GOOD")
        elif fen_accuracy >= 40:
            print("⚠️  FEN generation: FAIR")
        else:
            print("❌ FEN generation: NEEDS IMPROVEMENT")
        
        # Per-image analysis
        print(f"\nPER-IMAGE ANALYSIS:")
        print("-" * 40)
        for result in self.results:
            img_metrics = result['metrics']
            img_occ_acc = (img_metrics['occupancy_correct'] / img_metrics['occupancy_total']) * 100
            img_color_acc = (img_metrics['color_correct'] / img_metrics['color_total']) * 100 if img_metrics['color_total'] > 0 else 0
            img_piece_acc = (img_metrics['piece_correct'] / img_metrics['piece_total']) * 100 if img_metrics['piece_total'] > 0 else 0
            
            print(f"{result['image']}:")
            print(f"  Occupancy: {img_occ_acc:.1f}% ({img_metrics['occupancy_correct']}/{img_metrics['occupancy_total']})")
            print(f"  Color: {img_color_acc:.1f}% ({img_metrics['color_correct']}/{img_metrics['color_total']})")
            print(f"  Piece: {img_piece_acc:.1f}% ({img_metrics['piece_correct']}/{img_metrics['piece_total']})")
            print(f"  FEN Perfect: {'Yes' if result['fen_match'] else 'No'}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed results"""
        results_file = "comprehensive_accuracy_results.json"
        
        # Calculate percentages
        occupancy_accuracy = (self.metrics['occupancy_correct'] / self.metrics['occupancy_total']) * 100
        color_accuracy = (self.metrics['color_correct'] / self.metrics['color_total']) * 100 if self.metrics['color_total'] > 0 else 0
        piece_accuracy = (self.metrics['piece_correct'] / self.metrics['piece_total']) * 100 if self.metrics['piece_total'] > 0 else 0
        fen_accuracy = (self.metrics['fen_perfect'] / self.metrics['images_total']) * 100
        
        save_data = {
            'summary': {
                'images_evaluated': self.metrics['images_total'],
                'occupancy_accuracy': occupancy_accuracy,
                'color_accuracy': color_accuracy,
                'piece_accuracy': piece_accuracy,
                'fen_accuracy': fen_accuracy,
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
    print("Comprehensive Chess Model Accuracy Evaluation")
    print("=" * 50)
    
    evaluator = ComprehensiveAccuracyEvaluator()
    
    if not evaluator.check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Check if ground truth annotations exist
    dataset_path = "my_chess_images/train/images"
    
    # Look for existing annotations
    annotation_files = list(Path(dataset_path).glob("**/*.json"))
    
    if not annotation_files:
        print("No ground truth annotations found!")
        print("Please create annotations first:")
        print("  python create_ground_truth_interactive.py")
        return
    
    print(f"Found {len(annotation_files)} annotation files")
    
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
