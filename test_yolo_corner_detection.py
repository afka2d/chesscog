#!/usr/bin/env python3
"""
Test the trained YOLO corner detection model.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOCornerDetector:
    """YOLO-based corner detection service"""
    
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
            logger.info(f"YOLO chessboard detection model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_corners(self, image_path):
        """Detect corners using YOLO"""
        if self.model is None:
            return None
        
        try:
            # Run YOLO inference
            results = self.model(image_path, verbose=False)
            
            if not results or len(results) == 0:
                logger.warning("No detections found")
                return None
            
            result = results[0]
            
            # Check if we have segmentation masks
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                logger.info("Using segmentation masks for corner extraction")
                return self._extract_corners_from_masks(result)
            
            # Fallback to bounding boxes
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                logger.info("Using bounding boxes for corner extraction")
                return self._extract_corners_from_boxes(result, image_path)
            
            else:
                logger.warning("No usable detections found")
                return None
                
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return None
    
    def _extract_corners_from_masks(self, result):
        """Extract corners from segmentation masks"""
        try:
            # Get the detection with highest confidence
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            logger.info(f"Best detection confidence: {confidences[best_idx]:.3f}")
            
            # Get the mask polygon
            mask_coords = result.masks.xy[best_idx]
            
            if len(mask_coords) < 4:
                logger.warning("Insufficient mask points")
                return None
            
            # Convert polygon to 4 corners
            corners = self._polygon_to_corners(mask_coords)
            
            if corners is not None:
                logger.info(f"Extracted {len(corners)} corners from mask")
                return corners.tolist()
            else:
                logger.warning("Failed to extract corners from mask")
                return None
                
        except Exception as e:
            logger.error(f"Mask extraction failed: {e}")
            return None
    
    def _extract_corners_from_boxes(self, result, image_path):
        """Extract corners from bounding boxes (fallback)"""
        try:
            # Get the box with highest confidence
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            x1, y1, x2, y2 = boxes[best_idx]
            
            logger.info(f"Using bounding box: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
            
            # Convert bounding box to corner coordinates
            corners = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]   # Bottom-left
            ])
            
            return corners.tolist()
            
        except Exception as e:
            logger.error(f"Box extraction failed: {e}")
            return None
    
    def _polygon_to_corners(self, polygon_points):
        """Convert polygon points to 4 corners"""
        try:
            if len(polygon_points) < 4:
                return None
            
            # Find convex hull to get outer boundary
            hull = cv2.convexHull(polygon_points.astype(np.float32))
            hull_points = hull.reshape(-1, 2)
            
            if len(hull_points) < 4:
                return None
            
            # Approximate polygon to quadrilateral
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(-1, 2)
                logger.info("Approximated polygon to 4 corners")
            else:
                # Find 4 extreme points
                logger.info(f"Polygon has {len(approx)} points, finding 4 extreme points")
                corners = self._find_extreme_corners(hull_points)
            
            # Order corners consistently
            ordered_corners = self._order_corners(corners)
            
            return ordered_corners
            
        except Exception as e:
            logger.error(f"Polygon to corners conversion failed: {e}")
            return None
    
    def _find_extreme_corners(self, points):
        """Find 4 extreme corners from point cloud"""
        # Find extreme points
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
        return corners
    
    def _order_corners(self, corners):
        """Order corners consistently (clockwise from top-left)"""
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        ordered_corners = corners[sorted_indices]
        
        return ordered_corners

def test_yolo_model():
    """Test the YOLO corner detection model"""
    print("🧪 TESTING YOLO CORNER DETECTION MODEL")
    print("=" * 60)
    
    # Initialize YOLO service
    yolo_service = YOLOCornerDetector()
    
    if yolo_service.model is None:
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
        
        # Test YOLO detection
        start_time = time.time()
        yolo_corners = yolo_service.detect_corners(image_path)
        inference_time = time.time() - start_time
        
        if yolo_corners:
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
            
            print(f"   ✅ YOLO detection successful")
            print(f"   📊 Average error: {avg_error:.1f} pixels")
            print(f"   ⚡ Inference time: {inference_time:.3f} seconds")
            print(f"   📊 Per-corner errors: {[f'{e:.1f}' for e in errors]} pixels")
            print(f"   📍 Detected corners: {np.array(yolo_corners).astype(int).tolist()}")
            
        else:
            print(f"   ❌ YOLO detection failed")
            print(f"   ⚡ Inference time: {inference_time:.3f} seconds")
    
    # Summary
    if yolo_results:
        avg_error = np.mean([r['avg_error'] for r in yolo_results])
        avg_time = np.mean([r['inference_time'] for r in yolo_results])
        
        print(f"\n📊 YOLO MODEL PERFORMANCE SUMMARY:")
        print(f"   Successful detections: {len(yolo_results)}/{len(test_cases)}")
        print(f"   Average error: {avg_error:.1f} pixels")
        print(f"   Average inference time: {avg_time:.3f} seconds")
        
        # Compare with known CNN performance
        cnn_avg_error = 64.0  # Original CNN
        opt_avg_error = 60.0  # Optimized CNN
        
        print(f"\n📊 COMPARISON WITH CNN MODELS:")
        print(f"   Original CNN: {cnn_avg_error:.1f} pixels")
        print(f"   Optimized CNN: {opt_avg_error:.1f} pixels")
        print(f"   YOLO: {avg_error:.1f} pixels")
        
        # Determine best model
        if avg_error < opt_avg_error:
            improvement_vs_opt = ((opt_avg_error - avg_error) / opt_avg_error) * 100
            print(f"   🎯 YOLO WINS! {improvement_vs_opt:.1f}% better than Optimized CNN")
        elif avg_error < cnn_avg_error:
            improvement_vs_cnn = ((cnn_avg_error - avg_error) / cnn_avg_error) * 100
            print(f"   ✅ YOLO better than Original CNN ({improvement_vs_cnn:.1f}% improvement)")
            print(f"   ⚠️  But Optimized CNN still better ({opt_avg_error:.1f}px vs {avg_error:.1f}px)")
        else:
            print(f"   ⚠️  CNN models still perform better")
        
        # Performance assessment
        if avg_error < 30:
            print(f"   🎯 EXCELLENT: YOLO achieved target accuracy!")
        elif avg_error < 50:
            print(f"   ✅ VERY GOOD: YOLO performance is strong")
        elif avg_error < 70:
            print(f"   ✅ GOOD: YOLO performance is acceptable")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: YOLO needs tuning")
        
        return yolo_results
    else:
        print("❌ No successful YOLO detections")
        return None

def create_yolo_visualization():
    """Create visualization of YOLO corner detection"""
    print(f"\n🎨 CREATING YOLO CORNER DETECTION VISUALIZATION")
    print("-" * 50)
    
    # Test on best performing image
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
    
    # Get YOLO predictions
    yolo_service = YOLOCornerDetector()
    yolo_corners = yolo_service.detect_corners(image_path)
    
    if not yolo_corners:
        print("❌ YOLO detection failed")
        return
    
    yolo_corners = np.array(yolo_corners)
    
    # Create visualization
    vis_image = image.copy()
    
    # Draw ground truth corners (green)
    for i, corner in enumerate(gt_corners):
        cv2.circle(vis_image, tuple(corner.astype(int)), 30, (0, 255, 0), -1)
        cv2.circle(vis_image, tuple(corner.astype(int)), 35, (255, 255, 255), 4)
        cv2.putText(vis_image, f'GT{i}', tuple(corner.astype(int) + [-25, -45]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Draw YOLO corners (yellow)
    for i, corner in enumerate(yolo_corners):
        cv2.circle(vis_image, tuple(corner.astype(int)), 25, (0, 255, 255), -1)
        cv2.circle(vis_image, tuple(corner.astype(int)), 30, (0, 0, 0), 3)
        cv2.putText(vis_image, f'YOLO{i}', tuple(corner.astype(int) + [30, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw board outlines
    gt_corners_int = gt_corners.astype(np.int32)
    yolo_corners_int = yolo_corners.astype(np.int32)
    
    cv2.polylines(vis_image, [gt_corners_int], True, (0, 255, 0), 5)    # Green for GT
    cv2.polylines(vis_image, [yolo_corners_int], True, (0, 255, 255), 3)  # Yellow for YOLO
    
    # Calculate and display accuracy
    errors = np.sqrt(np.sum((gt_corners - yolo_corners) ** 2, axis=1))
    avg_error = np.mean(errors)
    
    # Add title and accuracy
    cv2.putText(vis_image, "YOLO CORNER DETECTION", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(vis_image, f"Average Error: {avg_error:.1f} pixels", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add legend
    h, w = image.shape[:2]
    cv2.putText(vis_image, "Green = Ground Truth, Yellow = YOLO", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Save visualization
    output_path = "yolo_corner_detection_result.jpg"
    cv2.imwrite(output_path, vis_image)
    
    print(f"✅ YOLO visualization saved: {output_path}")
    print(f"   Average error: {avg_error:.1f} pixels")

def compare_yolo_with_existing_models():
    """Compare YOLO with existing CNN models"""
    print(f"\n🏁 YOLO vs CNN CORNER DETECTION COMPARISON")
    print("=" * 60)
    
    # Initialize all services
    services = {}
    
    try:
        from corner_detection_service import CornerDetectionService
        services['Original CNN'] = CornerDetectionService()
        print("✅ Original CNN loaded")
    except:
        print("❌ Original CNN failed to load")
    
    try:
        from optimized_corner_service import OptimizedCornerService
        services['Optimized CNN'] = OptimizedCornerService()
        print("✅ Optimized CNN loaded")
    except:
        print("❌ Optimized CNN failed to load")
    
    try:
        services['YOLO'] = YOLOCornerDetector()
        if services['YOLO'].model is None:
            del services['YOLO']
            print("❌ YOLO model not available")
        else:
            print("✅ YOLO loaded")
    except:
        print("❌ YOLO failed to load")
    
    if len(services) < 2:
        print("❌ Need at least 2 services for comparison")
        return
    
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
    
    # Run comparison
    service_results = {name: [] for name in services.keys()}
    
    for test_case in test_cases:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        image_name = Path(image_path).name
        print(f"\n📸 Testing: {image_name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Test each service
        for service_name, service in services.items():
            start_time = time.time()
            corners = service.detect_corners(image_path)
            inference_time = time.time() - start_time
            
            if corners:
                corners_array = np.array(corners)
                errors = np.sqrt(np.sum((gt_corners - corners_array) ** 2, axis=1))
                avg_error = np.mean(errors)
                
                service_results[service_name].append({
                    'image': image_name,
                    'error': avg_error,
                    'time': inference_time
                })
                
                print(f"   {service_name}: {avg_error:.1f}px ({inference_time:.3f}s)")
            else:
                print(f"   {service_name}: ❌ Failed ({inference_time:.3f}s)")
    
    # Calculate final statistics
    print(f"\n📊 FINAL COMPARISON RESULTS:")
    print("=" * 40)
    
    final_stats = {}
    for service_name, results in service_results.items():
        if results:
            avg_error = np.mean([r['error'] for r in results])
            avg_time = np.mean([r['time'] for r in results])
            success_rate = len(results) / len(test_cases) * 100
            
            final_stats[service_name] = {
                'avg_error': avg_error,
                'avg_time': avg_time,
                'success_rate': success_rate
            }
            
            print(f"\n🎯 {service_name.upper()}:")
            print(f"   Average error: {avg_error:.1f} pixels")
            print(f"   Average time: {avg_time:.3f} seconds")
            print(f"   Success rate: {success_rate:.0f}%")
    
    # Determine winner
    if final_stats:
        # Sort by accuracy
        sorted_by_accuracy = sorted(final_stats.items(), key=lambda x: x[1]['avg_error'])
        
        print(f"\n🏆 ACCURACY RANKING:")
        for i, (service_name, stats) in enumerate(sorted_by_accuracy):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            print(f"   {rank_emoji} {service_name}: {stats['avg_error']:.1f}px")
        
        # Best overall recommendation
        best_service, best_stats = sorted_by_accuracy[0]
        print(f"\n🎯 BEST CORNER DETECTION METHOD: {best_service}")
        print(f"   Accuracy: {best_stats['avg_error']:.1f} pixels")
        print(f"   Speed: {best_stats['avg_time']:.3f} seconds")
        print(f"   Reliability: {best_stats['success_rate']:.0f}%")
        
        return final_stats
    
    return None

def main():
    """Main testing function"""
    print("YOLO Corner Detection Testing")
    print("=" * 50)
    
    # Test YOLO model
    yolo_results = test_yolo_model()
    
    if yolo_results:
        # Create visualization
        create_yolo_visualization()
        
        # Compare with existing models
        comparison_stats = compare_yolo_with_existing_models()
        
        print(f"\n🎯 TESTING COMPLETE!")
        print("   YOLO model is working and has been compared with CNN models")
        print("   Check yolo_corner_detection_result.jpg for visual results")
    else:
        print("❌ YOLO testing failed")
        print("   Model may still be training or there was an error")

if __name__ == "__main__":
    main()
