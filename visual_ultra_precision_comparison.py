#!/usr/bin/env python3
"""
Visual Ultra Precision Corner Comparison
========================================

Create visual comparisons showing the accuracy improvements of the
Ultra Precision API vs existing methods, with ground truth overlay.
"""

import requests
import json
import cv2
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualUltraPrecisionComparison:
    def __init__(self):
        self.apis = {
            'YOLO Only (Port 8002)': 'http://localhost:8002',
            'Fast Precision (Port 8004)': 'http://localhost:8004',
            'Ultra Precision (Port 8005)': 'http://localhost:8005'
        }
        
        self.test_cases = [
            {
                'image': 'my_chess_images/train/images/IMG_4698.JPG',
                'annotation': 'my_chess_images/train/annotations/IMG_4698.json'
            }
        ]
    
    def create_comprehensive_comparison(self):
        """
        Create comprehensive visual comparison of all corner detection methods
        """
        logger.info("🎯 CREATING ULTRA PRECISION VISUAL COMPARISON")
        logger.info("=" * 60)
        
        for test_case in self.test_cases:
            image_path = test_case['image']
            annotation_path = test_case['annotation']
            
            if not Path(image_path).exists():
                logger.warning(f"⚠️  Image not found: {image_path}")
                continue
            
            if not Path(annotation_path).exists():
                logger.warning(f"⚠️  Annotation not found: {annotation_path}")
                continue
            
            logger.info(f"📸 Processing: {Path(image_path).name}")
            
            # Load ground truth
            ground_truth_corners = self._load_ground_truth(annotation_path)
            if ground_truth_corners is None:
                logger.warning("⚠️  Could not load ground truth corners")
                continue
            
            # Test all APIs
            api_results = {}
            for api_name, api_url in self.apis.items():
                try:
                    if 'Ultra Precision' in api_name:
                        result = self._test_ultra_precision_api(api_url, image_path, time_budget=2.0)
                    else:
                        result = self._test_standard_api(api_url, image_path)
                    
                    if result['success']:
                        api_results[api_name] = result
                        logger.info(f"   {api_name}: ✅ {result['time_taken']:.3f}s")
                    else:
                        logger.warning(f"   {api_name}: ❌ Failed")
                        
                except Exception as e:
                    logger.warning(f"   {api_name}: ❌ Error: {e}")
            
            # Create visual comparison
            if api_results:
                self._create_side_by_side_comparison(
                    image_path, ground_truth_corners, api_results
                )
                
                # Calculate and display accuracy metrics
                self._calculate_accuracy_metrics(ground_truth_corners, api_results)
    
    def _load_ground_truth(self, annotation_path: str):
        """
        Load ground truth corners from annotation file
        """
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
            # Extract corners
            if 'corners' in data:
                return data['corners']
            else:
                logger.warning(f"No 'corners' key found in {annotation_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return None
    
    def _test_ultra_precision_api(self, api_url: str, image_path: str, time_budget: float):
        """
        Test Ultra Precision API
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{api_url}/detect_corners",
                files=files,
                params={'time_budget': time_budget},
                timeout=time_budget + 5
            )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': data.get('processing_time'),
                'budget_met': data.get('budget_met'),
                'features_used': data.get('features_used', [])
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _test_standard_api(self, api_url: str, image_path: str):
        """
        Test standard APIs
        """
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect_corners", files=files, timeout=10)
        
        time_taken = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': time_taken
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _create_side_by_side_comparison(self, image_path: str, ground_truth_corners, api_results):
        """
        Create side-by-side visual comparison
        """
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Calculate number of subplots needed
        num_methods = len(api_results) + 1  # +1 for ground truth
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'🎯 Ultra Precision Corner Detection Comparison\n{Path(image_path).name}', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot ground truth
        axes[0].imshow(original_img)
        self._draw_corners_on_plot(axes[0], ground_truth_corners, 'Ground Truth', 'green')
        
        # Plot API results
        colors = ['red', 'blue', 'orange', 'purple']
        
        for idx, (api_name, result) in enumerate(api_results.items()):
            if idx >= 3:  # Max 3 API results to fit in 2x2 grid
                break
            
            axes[idx + 1].imshow(original_img)
            
            if result['success']:
                color = colors[idx % len(colors)]
                self._draw_corners_on_plot(axes[idx + 1], result['corners'], api_name, color)
                
                # Calculate error vs ground truth
                error = self._calculate_corner_error(ground_truth_corners, result['corners'])
                
                # Add performance info
                time_info = f"Time: {result.get('time_taken', 0):.3f}s"
                error_info = f"Error: {error:.1f}px"
                
                if 'Ultra Precision' in api_name:
                    budget_info = f"Budget: {'✅' if result.get('budget_met', True) else '❌'}"
                    title = f"{api_name}\n{time_info} | {error_info} | {budget_info}"
                else:
                    title = f"{api_name}\n{time_info} | {error_info}"
                
                axes[idx + 1].set_title(title, fontsize=10)
            else:
                axes[idx + 1].set_title(f"{api_name}\n❌ Failed", fontsize=10, color='red')
        
        # Hide unused subplot
        if len(api_results) < 3:
            axes[3].set_visible(False)
        
        # Save comparison
        output_path = f"ultra_precision_comparison_{Path(image_path).stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visual comparison saved: {output_path}")
        
        return output_path
    
    def _draw_corners_on_plot(self, ax, corners, title, color):
        """
        Draw corners on matplotlib plot
        """
        if not corners:
            ax.set_title(f"{title}\n❌ No corners detected", color='red')
            return
        
        corners_np = np.array(corners)
        
        # Draw corner points
        ax.scatter(corners_np[:, 0], corners_np[:, 1], c=color, s=100, alpha=0.8)
        
        # Draw quadrilateral
        quad = plt.Polygon(corners_np, fill=False, edgecolor=color, linewidth=3, alpha=0.8)
        ax.add_patch(quad)
        
        # Label corners
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, label) in enumerate(zip(corners_np, labels)):
            ax.annotate(label, (corner[0], corner[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', color=color)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, ax.get_images()[0].get_array().shape[1])
        ax.set_ylim(ax.get_images()[0].get_array().shape[0], 0)
    
    def _calculate_corner_error(self, ground_truth, predicted):
        """
        Calculate average pixel error between ground truth and predicted corners
        """
        if not ground_truth or not predicted:
            return float('inf')
        
        gt_np = np.array(ground_truth)
        pred_np = np.array(predicted)
        
        # Calculate Euclidean distance for each corner
        errors = np.linalg.norm(gt_np - pred_np, axis=1)
        return np.mean(errors)
    
    def _calculate_accuracy_metrics(self, ground_truth_corners, api_results):
        """
        Calculate and display accuracy metrics
        """
        logger.info("\n📊 ACCURACY METRICS:")
        logger.info("-" * 50)
        
        for api_name, result in api_results.items():
            if result['success']:
                error = self._calculate_corner_error(ground_truth_corners, result['corners'])
                time_taken = result.get('time_taken', 0)
                
                # Performance rating
                if error < 15:
                    rating = "🏆 EXCELLENT"
                elif error < 25:
                    rating = "✅ GOOD"
                elif error < 40:
                    rating = "⚠️  FAIR"
                else:
                    rating = "❌ POOR"
                
                logger.info(f"{api_name}:")
                logger.info(f"   Error: {error:.1f}px {rating}")
                logger.info(f"   Time: {time_taken:.3f}s")
                
                if 'Ultra Precision' in api_name:
                    budget_met = result.get('budget_met', True)
                    features = result.get('features_used', [])
                    logger.info(f"   Budget Met: {'✅' if budget_met else '❌'}")
                    logger.info(f"   Features: {', '.join(features)}")
                
                logger.info("")

def main():
    """
    Run the visual comparison
    """
    comparator = VisualUltraPrecisionComparison()
    comparator.create_comprehensive_comparison()

if __name__ == "__main__":
    main()
