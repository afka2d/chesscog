#!/usr/bin/env python3
"""
Simple model accuracy evaluation using the local development API.
This evaluates real-world accuracy without requiring ground truth annotations.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.results = []
        
    def detect_chessboard_corners(self, image_path):
        """Detect chessboard corners using OpenCV"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Convert to the format expected by the API
            corners_2d = corners.reshape(-1, 2)
            
            # Find the 4 outer corners
            top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
            top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
            bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
            bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
            
            return [top_left, top_right, bottom_right, bottom_left]
        else:
            # Fallback: estimate corners based on image dimensions
            h, w = img.shape[:2]
            margin = min(h, w) * 0.1
            
            return [
                [margin, margin],
                [w - margin, margin],
                [w - margin, h - margin],
                [margin, h - margin]
            ]
    
    def evaluate_image(self, image_path):
        """Evaluate a single image using the API"""
        logger.info(f"Evaluating: {image_path}")
        
        try:
            # Detect corners
            corners = self.detect_chessboard_corners(image_path)
            
            # Call API with debug mode
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
                
                # Extract metrics
                pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
                occupied_squares = sum(result.get('occupancy', []))
                
                # Get debug info if available
                debug_info = result.get('debug_info', {})
                processing_time = debug_info.get('processing_time', 0)
                confidence_scores = debug_info.get('confidence_scores', [])
                
                # Calculate confidence statistics
                if confidence_scores:
                    occupancy_confidences = [s['occupancy_confidence'] for s in confidence_scores]
                    color_confidences = [s['color_confidence'] for s in confidence_scores]
                    piece_confidences = [s['piece_confidence'] for s in confidence_scores]
                    
                    conf_stats = {
                        'occupancy_mean': np.mean(occupancy_confidences),
                        'occupancy_std': np.std(occupancy_confidences),
                        'color_mean': np.mean(color_confidences),
                        'color_std': np.std(color_confidences),
                        'piece_mean': np.mean(piece_confidences),
                        'piece_std': np.std(piece_confidences)
                    }
                else:
                    conf_stats = {}
                
                image_result = {
                    'image_path': str(image_path),
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'processing_time': processing_time,
                    'confidence_stats': conf_stats,
                    'fen': result.get('fen', ''),
                    'success': result.get('success', False)
                }
                
                self.results.append(image_result)
                
                logger.info(f"  Pieces detected: {pieces_detected}")
                logger.info(f"  Occupied squares: {occupied_squares}")
                logger.info(f"  Processing time: {processing_time:.3f}s")
                
                if conf_stats:
                    logger.info(f"  Avg occupancy confidence: {conf_stats.get('occupancy_mean', 0):.3f}")
                    logger.info(f"  Avg color confidence: {conf_stats.get('color_mean', 0):.3f}")
                    logger.info(f"  Avg piece confidence: {conf_stats.get('piece_mean', 0):.3f}")
                
                return image_result
            else:
                logger.error(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating {image_path}: {e}")
            return None
    
    def evaluate_dataset(self, dataset_path):
        """Evaluate entire dataset"""
        logger.info(f"Starting evaluation of dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(images)} images to evaluate")
        
        # Evaluate each image
        for i, image_path in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}: {image_path.name}")
            
            image_result = self.evaluate_image(str(image_path))
            if image_result is None:
                continue
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate summary report of all evaluations"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION SUMMARY REPORT")
        logger.info("=" * 60)
        
        # Basic statistics
        total_images = len(self.results)
        successful_images = sum(1 for r in self.results if r['success'])
        
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Successful API calls: {successful_images}")
        logger.info(f"Success rate: {successful_images/total_images:.3f}")
        
        # Pieces detection statistics
        pieces_detected = [r['pieces_detected'] for r in self.results if r['success']]
        occupied_squares = [r['occupied_squares'] for r in self.results if r['success']]
        processing_times = [r['processing_time'] for r in self.results if r['success']]
        
        if pieces_detected:
            logger.info(f"\nPieces Detection:")
            logger.info(f"  Total pieces detected: {sum(pieces_detected)}")
            logger.info(f"  Average pieces per image: {np.mean(pieces_detected):.2f}")
            logger.info(f"  Min pieces in image: {min(pieces_detected)}")
            logger.info(f"  Max pieces in image: {max(pieces_detected)}")
        
        if occupied_squares:
            logger.info(f"\nOccupancy Detection:")
            logger.info(f"  Total occupied squares: {sum(occupied_squares)}")
            logger.info(f"  Average occupied per image: {np.mean(occupied_squares):.2f}")
            logger.info(f"  Min occupied in image: {min(occupied_squares)}")
            logger.info(f"  Max occupied in image: {max(occupied_squares)}")
        
        if processing_times:
            logger.info(f"\nPerformance:")
            logger.info(f"  Average processing time: {np.mean(processing_times):.3f}s")
            logger.info(f"  Min processing time: {min(processing_times):.3f}s")
            logger.info(f"  Max processing time: {max(processing_times):.3f}s")
        
        # Confidence statistics
        all_confidence_stats = [r['confidence_stats'] for r in self.results if r['success'] and r['confidence_stats']]
        
        if all_confidence_stats:
            logger.info(f"\nConfidence Analysis:")
            
            # Occupancy confidence
            occ_confidences = [s.get('occupancy_mean', 0) for s in all_confidence_stats if 'occupancy_mean' in s]
            if occ_confidences:
                logger.info(f"  Average occupancy confidence: {np.mean(occ_confidences):.3f} ± {np.std(occ_confidences):.3f}")
            
            # Color confidence
            color_confidences = [s.get('color_mean', 0) for s in all_confidence_stats if 'color_mean' in s]
            if color_confidences:
                logger.info(f"  Average color confidence: {np.mean(color_confidences):.3f} ± {np.std(color_confidences):.3f}")
            
            # Piece confidence
            piece_confidences = [s.get('piece_mean', 0) for s in all_confidence_stats if 'piece_mean' in s]
            if piece_confidences:
                logger.info(f"  Average piece confidence: {np.mean(piece_confidences):.3f} ± {np.std(piece_confidences):.3f}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed results to file"""
        results_file = "evaluation_results.json"
        
        save_data = {
            'summary': {
                'total_images': len(self.results),
                'successful_images': sum(1 for r in self.results if r['success']),
                'total_pieces_detected': sum(r['pieces_detected'] for r in self.results if r['success']),
                'total_occupied_squares': sum(r['occupied_squares'] for r in self.results if r['success']),
                'average_processing_time': np.mean([r['processing_time'] for r in self.results if r['success']]) if self.results else 0
            },
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {results_file}")

def main():
    """Main evaluation function"""
    # Check if local API is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            logger.error("Local development API is not running!")
            logger.info("Please start it with: ./start_local_dev.sh")
            return
    except:
        logger.error("Cannot connect to local development API!")
        logger.info("Please start it with: ./start_local_dev.sh")
        return
    
    evaluator = SimpleEvaluator()
    
    # Evaluate on your training images
    dataset_path = "my_chess_images/train/images"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        logger.info("Please ensure you have chess images in the correct directory")
        return
    
    evaluator.evaluate_dataset(dataset_path)

if __name__ == "__main__":
    main()
