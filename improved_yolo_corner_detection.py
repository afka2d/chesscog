#!/usr/bin/env python3
"""
Improved YOLO corner detection with better corner extraction logic.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedYOLOCornerDetector:
    """Improved YOLO-based corner detection with better extraction logic"""
    
    def __init__(self, model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"YOLO model not found: {self.model_path}")
                return False
            
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_corners(self, image_path, conf_threshold=0.05, visualize=False):
        """
        Detect corners using YOLO with improved extraction logic.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detection
            visualize: Whether to save visualization
        """
        if self.model is None:
            return None
        
        try:
            # Run YOLO inference with low confidence threshold
            results = self.model(image_path, verbose=False, conf=conf_threshold)
            
            if not results or len(results) == 0:
                logger.warning("No YOLO results")
                return None
            
            result = results[0]
            
            # Debug: print what we found
            boxes_count = len(result.boxes) if result.boxes else 0
            masks_count = len(result.masks) if result.masks else 0
            logger.info(f"YOLO found {boxes_count} boxes, {masks_count} masks")
            
            # Try masks first (more precise)
            if result.masks is not None and len(result.masks) > 0:
                corners = self._extract_corners_from_masks(result)
                if corners is not None:
                    logger.info("Successfully extracted corners from masks")
                    if visualize:
                        self._save_detection_visualization(image_path, result, corners)
                    return corners
            
            # Fallback to bounding boxes
            if result.boxes is not None and len(result.boxes) > 0:
                corners = self._extract_corners_from_boxes(result)
                if corners is not None:
                    logger.info("Successfully extracted corners from bounding boxes")
                    if visualize:
                        self._save_detection_visualization(image_path, result, corners)
                    return corners
            
            logger.warning("No corners could be extracted")
            return None
                
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return None
    
    def _extract_corners_from_masks(self, result):
        """Extract corners from segmentation masks with improved logic"""
        try:
            # Get all detections and their confidences
            confidences = result.boxes.conf.cpu().numpy()
            
            # Use detection with highest confidence
            best_idx = np.argmax(confidences)
            best_conf = confidences[best_idx]
            
            logger.info(f"Using detection {best_idx} with confidence {best_conf:.3f}")
            
            # Get mask points
            mask_points = result.masks.xy[best_idx]
            
            if len(mask_points) < 4:
                logger.warning(f"Mask has only {len(mask_points)} points")
                return None
            
            logger.info(f"Mask has {len(mask_points)} points")
            
            # Method 1: Try to approximate to quadrilateral
            corners = self._approximate_to_quadrilateral(mask_points)
            
            if corners is not None and len(corners) == 4:
                return self._order_corners(corners)
            
            # Method 2: Find 4 extreme points
            logger.info("Approximation failed, finding extreme points")
            corners = self._find_extreme_points(mask_points)
            
            if corners is not None and len(corners) == 4:
                return self._order_corners(corners)
            
            logger.warning("All corner extraction methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Mask corner extraction failed: {e}")
            return None
    
    def _extract_corners_from_boxes(self, result):
        """Extract corners from bounding boxes"""
        try:
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            # Use box with highest confidence
            best_idx = np.argmax(confidences)
            x1, y1, x2, y2 = boxes[best_idx]
            
            logger.info(f"Using bounding box: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
            
            # Create corners from bounding box
            corners = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]   # Bottom-left
            ])
            
            return corners
            
        except Exception as e:
            logger.error(f"Box corner extraction failed: {e}")
            return None
    
    def _approximate_to_quadrilateral(self, points):
        """Approximate polygon to quadrilateral"""
        try:
            # Convert to proper format
            points = points.astype(np.float32)
            
            # Find convex hull
            hull = cv2.convexHull(points)
            
            # Approximate to polygon with fewer points
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Try different epsilon values if we don't get 4 points
            for epsilon_factor in [0.02, 0.03, 0.05, 0.1]:
                if len(approx) == 4:
                    break
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                logger.info(f"Successfully approximated to quadrilateral")
                return approx.reshape(-1, 2)
            else:
                logger.info(f"Approximation resulted in {len(approx)} points")
                return None
                
        except Exception as e:
            logger.error(f"Quadrilateral approximation failed: {e}")
            return None
    
    def _find_extreme_points(self, points):
        """Find 4 extreme points from point cloud"""
        try:
            # Find extreme points in different directions
            top_idx = np.argmin(points[:, 1])
            bottom_idx = np.argmax(points[:, 1])
            left_idx = np.argmin(points[:, 0])
            right_idx = np.argmax(points[:, 0])
            
            # Get corner candidates
            top_left = points[np.argmin(points[:, 0] + points[:, 1])]
            top_right = points[np.argmax(points[:, 0] - points[:, 1])]
            bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
            bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
            
            corners = np.array([top_left, top_right, bottom_right, bottom_left])
            
            logger.info("Found 4 extreme points")
            return corners
            
        except Exception as e:
            logger.error(f"Extreme point finding failed: {e}")
            return None
    
    def _order_corners(self, corners):
        """Order corners consistently"""
        try:
            # Calculate center
            center = np.mean(corners, axis=0)
            
            # Calculate angles from center to each corner
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            
            # Sort by angle to get consistent ordering
            sorted_indices = np.argsort(angles)
            ordered_corners = corners[sorted_indices]
            
            return ordered_corners
            
        except Exception as e:
            logger.error(f"Corner ordering failed: {e}")
            return corners
    
    def _save_detection_visualization(self, image_path, result, corners):
        """Save visualization of YOLO detection"""
        try:
            image = cv2.imread(image_path)
            vis_image = image.copy()
            
            # Draw detected corners
            for i, corner in enumerate(corners):
                cv2.circle(vis_image, tuple(corner.astype(int)), 25, (0, 255, 255), -1)
                cv2.putText(vis_image, f'Y{i}', tuple(corner.astype(int) + [-20, -30]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw board outline
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis_image, [corners_int], True, (0, 255, 255), 4)
            
            # Add title
            cv2.putText(vis_image, "YOLO DETECTION", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            
            # Save
            output_path = f"yolo_detection_{Path(image_path).stem}.jpg"
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualization saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization saving failed: {e}")

def test_improved_yolo():
    """Test the improved YOLO corner detection"""
    print("🚀 TESTING IMPROVED YOLO CORNER DETECTION")
    print("=" * 60)
    
    # Initialize YOLO detector
    yolo_detector = ImprovedYOLOCornerDetector()
    
    if yolo_detector.model is None:
        print("❌ YOLO model not available")
        return False
    
    # Test cases
    test_cases = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json'
        }
    ]
    
    yolo_results = []
    
    for test_case in test_cases:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Test YOLO detection with visualization
        start_time = time.time()
        yolo_corners = yolo_detector.detect_corners(image_path, conf_threshold=0.05, visualize=True)
        inference_time = time.time() - start_time
        
        if yolo_corners is not None:
            yolo_corners_array = np.array(yolo_corners)
            errors = np.sqrt(np.sum((gt_corners - yolo_corners_array) ** 2, axis=1))
            avg_error = np.mean(errors)
            
            result = {
                'image': Path(image_path).name,
                'avg_error': avg_error,
                'per_corner_errors': errors.tolist(),
                'inference_time': inference_time,
                'corners': yolo_corners,
                'ground_truth': gt_corners.tolist()
            }
            yolo_results.append(result)
            
            print(f"   ✅ YOLO detection successful!")
            print(f"   📊 Average error: {avg_error:.1f} pixels")
            print(f"   ⚡ Inference time: {inference_time:.3f} seconds")
            print(f"   📊 Per-corner errors: {[f'{e:.1f}' for e in errors]} pixels")
            print(f"   📍 Ground truth: {gt_corners.astype(int).tolist()}")
            print(f"   🤖 YOLO detected: {yolo_corners_array.astype(int).tolist()}")
            
        else:
            print(f"   ❌ YOLO detection failed")
            print(f"   ⚡ Inference time: {inference_time:.3f} seconds")
    
    # Compare with existing models
    if yolo_results:
        print(f"\n📊 YOLO PERFORMANCE SUMMARY:")
        avg_error = np.mean([r['avg_error'] for r in yolo_results])
        avg_time = np.mean([r['inference_time'] for r in yolo_results])
        success_rate = len(yolo_results) / len(test_cases) * 100
        
        print(f"   Average error: {avg_error:.1f} pixels")
        print(f"   Average time: {avg_time:.3f} seconds")
        print(f"   Success rate: {success_rate:.0f}%")
        
        # Compare with known CNN performance
        print(f"\n📊 COMPARISON WITH EXISTING MODELS:")
        print(f"   Original CNN: 64.0 pixels")
        print(f"   Optimized CNN: 60.0 pixels")
        print(f"   YOLO: {avg_error:.1f} pixels")
        
        if avg_error < 60.0:
            improvement = ((60.0 - avg_error) / 60.0) * 100
            print(f"   🎯 YOLO WINS! {improvement:.1f}% better than Optimized CNN")
        elif avg_error < 64.0:
            improvement = ((64.0 - avg_error) / 64.0) * 100
            print(f"   ✅ YOLO better than Original CNN ({improvement:.1f}% improvement)")
        else:
            print(f"   ⚠️  CNN models still perform better")
        
        # Performance tier
        if avg_error < 30:
            print(f"   🎯 EXCELLENT: Production-ready accuracy!")
        elif avg_error < 50:
            print(f"   ✅ VERY GOOD: Suitable for automatic detection")
        elif avg_error < 70:
            print(f"   ✅ GOOD: Acceptable performance")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT")
        
        return yolo_results
    else:
        print(f"\n❌ No successful YOLO detections")
        return None

def create_comprehensive_comparison():
    """Create comprehensive comparison of all corner detection methods"""
    print(f"\n🏁 COMPREHENSIVE CORNER DETECTION COMPARISON")
    print("=" * 60)
    
    # Test image
    image_path = 'grey_background_dataset/images/test/IMG_4785.JPG'
    annotation_path = 'grey_background_dataset/annotations/test/IMG_4785.json'
    
    if not Path(image_path).exists() or not Path(annotation_path).exists():
        print("❌ Test files not found")
        return
    
    # Load image and ground truth
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Get predictions from all available methods
    methods = {}
    
    # Original CNN
    try:
        from corner_detection_service import CornerDetectionService
        cnn_service = CornerDetectionService()
        cnn_corners = cnn_service.detect_corners(image_path)
        if cnn_corners:
            methods['Original CNN'] = {
                'corners': np.array(cnn_corners),
                'color': (0, 0, 255),  # Red
                'error': np.mean(np.sqrt(np.sum((gt_corners - np.array(cnn_corners)) ** 2, axis=1)))
            }
    except:
        pass
    
    # Optimized CNN
    try:
        from optimized_corner_service import OptimizedCornerService
        opt_service = OptimizedCornerService()
        opt_corners = opt_service.detect_corners(image_path)
        if opt_corners:
            methods['Optimized CNN'] = {
                'corners': np.array(opt_corners),
                'color': (255, 0, 0),  # Blue
                'error': np.mean(np.sqrt(np.sum((gt_corners - np.array(opt_corners)) ** 2, axis=1)))
            }
    except:
        pass
    
    # YOLO
    try:
        yolo_service = ImprovedYOLOCornerDetector()
        yolo_corners = yolo_service.detect_corners(image_path, visualize=True)
        if yolo_corners:
            methods['YOLO'] = {
                'corners': np.array(yolo_corners),
                'color': (0, 255, 255),  # Yellow
                'error': np.mean(np.sqrt(np.sum((gt_corners - np.array(yolo_corners)) ** 2, axis=1)))
            }
    except:
        pass
    
    if not methods:
        print("❌ No methods available for comparison")
        return
    
    # Create comprehensive visualization
    comparison_image = image.copy()
    
    # Draw ground truth (green, largest)
    for i, corner in enumerate(gt_corners):
        cv2.circle(comparison_image, tuple(corner.astype(int)), 35, (0, 255, 0), -1)
        cv2.circle(comparison_image, tuple(corner.astype(int)), 40, (255, 255, 255), 4)
        cv2.putText(comparison_image, f'GT{i}', tuple(corner.astype(int) + [-30, -50]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Draw each method's corners
    for method_name, method_data in methods.items():
        corners = method_data['corners']
        color = method_data['color']
        error = method_data['error']
        
        # Draw corners
        for i, corner in enumerate(corners):
            cv2.circle(comparison_image, tuple(corner.astype(int)), 25, color, -1)
            cv2.circle(comparison_image, tuple(corner.astype(int)), 30, (255, 255, 255), 2)
        
        # Draw board outline
        corners_int = corners.astype(np.int32)
        cv2.polylines(comparison_image, [corners_int], True, color, 3)
    
    # Add legend and results
    h, w = image.shape[:2]
    legend_y = 120
    cv2.putText(comparison_image, "CORNER DETECTION COMPARISON", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    y_offset = 0
    for method_name, method_data in methods.items():
        color = method_data['color']
        error = method_data['error']
        cv2.putText(comparison_image, f"{method_name}: {error:.1f}px", (50, legend_y + 40 + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        y_offset += 40
    
    # Save comparison
    output_path = "comprehensive_corner_comparison.jpg"
    cv2.imwrite(output_path, comparison_image)
    
    print(f"✅ Comprehensive comparison saved: {output_path}")
    
    # Print ranking
    sorted_methods = sorted(methods.items(), key=lambda x: x[1]['error'])
    print(f"\n🏆 ACCURACY RANKING:")
    for i, (method_name, method_data) in enumerate(sorted_methods):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        print(f"   {rank_emoji} {method_name}: {method_data['error']:.1f} pixels")

def main():
    """Main testing function"""
    print("Improved YOLO Corner Detection Testing")
    print("=" * 50)
    
    # Test improved YOLO
    yolo_results = test_improved_yolo()
    
    if yolo_results:
        # Create comprehensive comparison
        create_comprehensive_comparison()
        
        print(f"\n🎯 YOLO TESTING COMPLETE!")
        print("   Check the generated visualization files:")
        print("   • yolo_detection_*.jpg - Individual YOLO detections")
        print("   • comprehensive_corner_comparison.jpg - All methods compared")
    else:
        print("❌ YOLO testing incomplete")
        print("   Model may need more training or different configuration")

if __name__ == "__main__":
    main()
