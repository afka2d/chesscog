#!/usr/bin/env python3
"""
Comprehensive evaluation system for chess model accuracy.
This provides detailed analysis and actionable recommendations.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessModelEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        
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
                
                pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
                occupied_squares = sum(result.get('occupancy', []))
                fen = result.get('fen', '')
                debug_info = result.get('debug_info', {})
                
                # Analyze debug info for confidence scores
                confidence_analysis = self.analyze_debug_info(debug_info)
                
                image_result = {
                    'image': Path(image_path).name,
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'fen': fen,
                    'success': result.get('success', False),
                    'confidence_analysis': confidence_analysis,
                    'debug_info': debug_info
                }
                
                self.results.append(image_result)
                
                logger.info(f"  Pieces: {pieces_detected}, Occupied: {occupied_squares}")
                if confidence_analysis:
                    logger.info(f"  Avg occupancy confidence: {confidence_analysis.get('avg_occupancy', 0):.3f}")
                
                return image_result
            else:
                logger.error(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating {image_path}: {e}")
            return None
    
    def analyze_debug_info(self, debug_info):
        """Analyze debug info for confidence scores"""
        if not debug_info or 'square_details' not in debug_info:
            return None
        
        square_details = debug_info['square_details']
        
        occupancy_scores = []
        color_scores = []
        piece_scores = []
        
        for square in square_details:
            if 'occupancy_probs' in square:
                occ_probs = square['occupancy_probs']
                occupied_prob = occ_probs.get('occupied', 0)
                occupancy_scores.append(occupied_prob)
            
            if 'color_confidence' in square:
                color_scores.append(square['color_confidence'])
            
            if 'piece_confidence' in square:
                piece_scores.append(square['piece_confidence'])
        
        analysis = {}
        
        if occupancy_scores:
            analysis['avg_occupancy'] = np.mean(occupancy_scores)
            analysis['max_occupancy'] = np.max(occupancy_scores)
            analysis['min_occupancy'] = np.min(occupancy_scores)
            analysis['occupancy_scores'] = occupancy_scores
        
        if color_scores:
            analysis['avg_color'] = np.mean(color_scores)
            analysis['max_color'] = np.max(color_scores)
            analysis['min_color'] = np.min(color_scores)
        
        if piece_scores:
            analysis['avg_piece'] = np.mean(piece_scores)
            analysis['max_piece'] = np.max(piece_scores)
            analysis['min_piece'] = np.min(piece_scores)
        
        return analysis
    
    def evaluate_dataset(self, dataset_path):
        """Evaluate entire dataset"""
        logger.info(f"Evaluating dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            return
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        if not images:
            logger.error("No images found to evaluate")
            return
        
        logger.info(f"Found {len(images)} images to evaluate")
        
        # Evaluate each image
        for i, image_path in enumerate(images):
            print(f"\n--- Image {i+1}/{len(images)} ---")
            self.evaluate_image(str(image_path))
            time.sleep(0.5)  # Small delay
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Basic statistics
        total_images = len(self.results)
        successful_images = sum(1 for r in self.results if r['success'])
        images_with_pieces = sum(1 for r in self.results if r['pieces_detected'] > 0)
        
        total_pieces = sum(r['pieces_detected'] for r in self.results)
        total_occupied = sum(r['occupied_squares'] for r in self.results)
        
        print(f"Images evaluated: {total_images}")
        print(f"Successful API calls: {successful_images}")
        print(f"Images with pieces detected: {images_with_pieces}")
        print(f"Total pieces detected: {total_pieces}")
        print(f"Total occupied squares: {total_occupied}")
        
        # Accuracy analysis
        if images_with_pieces > 0:
            detection_rate = images_with_pieces / total_images
            print(f"Detection rate: {detection_rate:.1%}")
            
            if detection_rate < 0.5:
                print("❌ LOW DETECTION RATE - Model needs improvement")
            elif detection_rate < 0.8:
                print("⚠️  MODERATE DETECTION RATE - Model needs tuning")
            else:
                print("✅ GOOD DETECTION RATE - Model performing well")
        
        # Confidence analysis
        all_occupancy_scores = []
        for result in self.results:
            if result['confidence_analysis'] and 'occupancy_scores' in result['confidence_analysis']:
                all_occupancy_scores.extend(result['confidence_analysis']['occupancy_scores'])
        
        if all_occupancy_scores:
            print(f"\nOccupancy Confidence Analysis:")
            print(f"  Average: {np.mean(all_occupancy_scores):.3f}")
            print(f"  Min: {np.min(all_occupancy_scores):.3f}")
            print(f"  Max: {np.max(all_occupancy_scores):.3f}")
            print(f"  Std: {np.std(all_occupancy_scores):.3f}")
            
            # Threshold recommendations
            scores_above_05 = sum(1 for s in all_occupancy_scores if s > 0.5)
            scores_above_03 = sum(1 for s in all_occupancy_scores if s > 0.3)
            scores_above_02 = sum(1 for s in all_occupancy_scores if s > 0.2)
            
            print(f"\nThreshold Analysis:")
            print(f"  Scores > 0.5: {scores_above_05}/{len(all_occupancy_scores)} ({scores_above_05/len(all_occupancy_scores):.1%})")
            print(f"  Scores > 0.3: {scores_above_03}/{len(all_occupancy_scores)} ({scores_above_03/len(all_occupancy_scores):.1%})")
            print(f"  Scores > 0.2: {scores_above_02}/{len(all_occupancy_scores)} ({scores_above_02/len(all_occupancy_scores):.1%})")
            
            # Recommend optimal threshold
            if scores_above_03 > scores_above_05:
                print(f"\n💡 RECOMMENDATION: Lower threshold to 0.3 (detects {scores_above_03} vs {scores_above_05} squares)")
            elif scores_above_02 > scores_above_03:
                print(f"\n💡 RECOMMENDATION: Lower threshold to 0.2 (detects {scores_above_02} vs {scores_above_03} squares)")
            else:
                print(f"\n💡 RECOMMENDATION: Current threshold (0.5) seems appropriate")
        
        # Per-image analysis
        print(f"\nPer-Image Analysis:")
        for result in self.results:
            print(f"  {result['image']}: {result['pieces_detected']} pieces, {result['occupied_squares']} occupied")
            if result['confidence_analysis']:
                avg_occ = result['confidence_analysis'].get('avg_occupancy', 0)
                print(f"    Avg occupancy confidence: {avg_occ:.3f}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed results"""
        results_file = "comprehensive_evaluation_results.json"
        
        save_data = {
            'summary': {
                'total_images': len(self.results),
                'successful_images': sum(1 for r in self.results if r['success']),
                'images_with_pieces': sum(1 for r in self.results if r['pieces_detected'] > 0),
                'total_pieces_detected': sum(r['pieces_detected'] for r in self.results),
                'total_occupied_squares': sum(r['occupied_squares'] for r in self.results)
            },
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file}")

def main():
    """Main evaluation function"""
    print("Comprehensive Chess Model Evaluation")
    print("=" * 40)
    
    evaluator = ChessModelEvaluator()
    
    if not evaluator.check_api():
        print("Please start the local API first:")
        print("  ./start_local_dev.sh")
        return
    
    # Evaluate dataset
    dataset_path = "my_chess_images/train/images"
    evaluator.evaluate_dataset(dataset_path)

if __name__ == "__main__":
    main()
