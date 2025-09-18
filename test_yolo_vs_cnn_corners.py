#!/usr/bin/env python3
"""
Compare YOLO vs CNN corner detection performance.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOCornerService:
    """YOLO-based corner detection service"""
    
    def __init__(self, model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            if not Path(self.model_path).exists():
                logger.warning(f"YOLO model not found: {self.model_path}")
                # Try alternative paths
                alt_paths = [
                    "yolo_runs/chessboard_detection/weights/best.pt",
                    "runs/segment/train/weights/best.pt",
                    "yolov8n-seg.pt"  # Fallback to pre-trained
                ]
                
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        self.model_path = alt_path
                        logger.info(f"Using alternative model: {alt_path}")
                        break
                else:
                    logger.error("No YOLO model found")
                    return False
            
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
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
                return None
            
            result = results[0]
            
            # Try to extract corners from detection
            if hasattr(result, 'masks') and result.masks is not None:
                # Segmentation model - extract from mask
                return self._extract_corners_from_mask(result)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Detection model - extract from bounding box
                return self._extract_corners_from_bbox(result, image_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return None
    
    def _extract_corners_from_mask(self, result):
        """Extract corners from segmentation mask"""
        try:
            if len(result.masks) == 0:
                return None
            
            # Get the mask with highest confidence
            best_mask_idx = torch.argmax(result.boxes.conf).item()
            mask_coords = result.masks.xy[best_mask_idx]
            
            if len(mask_coords) < 4:
                return None
            
            # Convert to 4 corners
            corners = self._polygon_to_corners(mask_coords)
            return corners.tolist() if corners is not None else None
            
        except Exception as e:
            logger.error(f"Mask extraction failed: {e}")
            return None
    
    def _extract_corners_from_bbox(self, result, image_path):
        """Extract corners from bounding box (fallback method)"""
        try:
            if len(result.boxes) == 0:
                return None
            
            # Get the box with highest confidence
            best_box = result.boxes[0]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            
            # Convert bounding box to corner coordinates
            corners = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]   # Bottom-left
            ])
            
            return corners.tolist()
            
        except Exception as e:
            logger.error(f"Bbox extraction failed: {e}")
            return None
    
    def _polygon_to_corners(self, polygon_points):
        """Convert polygon to 4 corners"""
        if len(polygon_points) < 4:
            return None
        
        # Find convex hull
        hull = cv2.convexHull(polygon_points.astype(np.float32))
        
        # Approximate to 4 corners
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
        else:
            # Fallback: find 4 extreme points
            hull_points = hull.reshape(-1, 2)
            center = np.mean(hull_points, axis=0)
            
            # Find corners by angle
            angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            
            # Select 4 evenly distributed points
            n = len(hull_points)
            corner_indices = [
                sorted_indices[0],
                sorted_indices[n // 4],
                sorted_indices[n // 2],
                sorted_indices[3 * n // 4]
            ]
            
            corners = hull_points[corner_indices]
        
        return corners

def compare_all_corner_detection_methods():
    """Compare YOLO vs CNN vs Optimized corner detection"""
    print("🏁 COMPREHENSIVE CORNER DETECTION COMPARISON")
    print("=" * 60)
    
    # Initialize all services
    services = {}
    
    # CNN-based services
    try:
        from corner_detection_service import CornerDetectionService
        services['Original CNN'] = CornerDetectionService()
        print("✅ Original CNN service loaded")
    except Exception as e:
        print(f"❌ Original CNN service failed: {e}")
    
    try:
        from optimized_corner_service import OptimizedCornerService
        services['Optimized CNN'] = OptimizedCornerService()
        print("✅ Optimized CNN service loaded")
    except Exception as e:
        print(f"❌ Optimized CNN service failed: {e}")
    
    # YOLO service
    try:
        yolo_service = YOLOCornerService()
        if yolo_service.model is not None:
            services['YOLO'] = yolo_service
            print("✅ YOLO service loaded")
        else:
            print("⚠️  YOLO service not available (model not trained yet)")
    except Exception as e:
        print(f"❌ YOLO service failed: {e}")
    
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
    
    results = {}
    
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
        
        results[image_name] = {'ground_truth': gt_corners.tolist()}
        
        # Test each service
        for service_name, service in services.items():
            start_time = time.time()
            
            try:
                corners = service.detect_corners(image_path)
                inference_time = time.time() - start_time
                
                if corners:
                    corners_array = np.array(corners)
                    errors = np.sqrt(np.sum((gt_corners - corners_array) ** 2, axis=1))
                    avg_error = np.mean(errors)
                    
                    results[image_name][service_name] = {
                        'corners': corners,
                        'avg_error': avg_error,
                        'per_corner_errors': errors.tolist(),
                        'inference_time': inference_time
                    }
                    
                    print(f"   {service_name}: {avg_error:.1f}px ({inference_time:.3f}s)")
                else:
                    results[image_name][service_name] = {
                        'corners': None,
                        'avg_error': float('inf'),
                        'inference_time': inference_time
                    }
                    print(f"   {service_name}: ❌ Failed ({inference_time:.3f}s)")
                    
            except Exception as e:
                print(f"   {service_name}: ❌ Error - {e}")
                results[image_name][service_name] = {
                    'corners': None,
                    'avg_error': float('inf'),
                    'inference_time': 0
                }
    
    # Calculate overall statistics
    print(f"\n📊 OVERALL COMPARISON SUMMARY:")
    print("=" * 40)
    
    service_stats = {}
    for service_name in services.keys():
        errors = []
        times = []
        success_count = 0
        
        for image_name, image_results in results.items():
            if service_name in image_results:
                result = image_results[service_name]
                if result['avg_error'] != float('inf'):
                    errors.append(result['avg_error'])
                    success_count += 1
                times.append(result['inference_time'])
        
        if errors:
            avg_error = np.mean(errors)
            avg_time = np.mean(times)
            
            service_stats[service_name] = {
                'avg_error': avg_error,
                'avg_time': avg_time,
                'success_rate': success_count / len(results) * 100
            }
            
            print(f"\n🎯 {service_name.upper()}:")
            print(f"   Average error: {avg_error:.1f} pixels")
            print(f"   Average time: {avg_time:.3f} seconds")
            print(f"   Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
        else:
            print(f"\n❌ {service_name.upper()}: No successful detections")
    
    # Determine best method
    if service_stats:
        # Rank by accuracy (lower error is better)
        sorted_services = sorted(service_stats.items(), key=lambda x: x[1]['avg_error'])
        
        print(f"\n🏆 RANKING BY ACCURACY:")
        for i, (service_name, stats) in enumerate(sorted_services):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            print(f"   {rank_emoji} {service_name}: {stats['avg_error']:.1f}px ({stats['avg_time']:.3f}s)")
        
        # Best service recommendation
        best_service, best_stats = sorted_services[0]
        print(f"\n🎯 RECOMMENDED SERVICE: {best_service}")
        print(f"   Best accuracy: {best_stats['avg_error']:.1f} pixels")
        print(f"   Speed: {best_stats['avg_time']:.3f} seconds")
        print(f"   Reliability: {best_stats['success_rate']:.0f}% success rate")
        
        # Performance tier assessment
        if best_stats['avg_error'] < 30:
            print("   🎯 EXCELLENT: Production-ready accuracy!")
        elif best_stats['avg_error'] < 50:
            print("   ✅ VERY GOOD: Suitable for automatic detection")
        elif best_stats['avg_error'] < 70:
            print("   ✅ GOOD: Acceptable for most use cases")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Consider further optimization")
    
    return results, service_stats

def check_yolo_training_status():
    """Check if YOLO training is complete"""
    print("🔍 CHECKING YOLO TRAINING STATUS")
    print("=" * 40)
    
    # Possible model locations
    model_paths = [
        "yolo_training_runs/yolo_chessboard_v1/weights/best.pt",
        "yolo_runs/chessboard_detection/weights/best.pt",
        "runs/segment/train/weights/best.pt"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"✅ YOLO model found: {model_path}")
            
            # Check model file size and modification time
            stat = Path(model_path).stat()
            size_mb = stat.st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Modified: {time.ctime(stat.st_mtime)}")
            
            return True, model_path
    
    print("⏳ YOLO training still in progress or not started")
    print("   Check background process or run: python train_yolo_chessboard.py")
    
    return False, None

def create_yolo_corner_visualization():
    """Create visualization comparing YOLO vs CNN corner detection"""
    print(f"\n🎨 CREATING YOLO vs CNN VISUALIZATION")
    print("-" * 40)
    
    # Check if YOLO model is ready
    yolo_ready, yolo_model_path = check_yolo_training_status()
    
    if not yolo_ready:
        print("⚠️  YOLO model not ready yet - skipping visualization")
        return False
    
    # Test image
    image_path = 'grey_background_dataset/images/test/IMG_4785.JPG'
    annotation_path = 'grey_background_dataset/annotations/test/IMG_4785.json'
    
    if not Path(image_path).exists() or not Path(annotation_path).exists():
        print("❌ Test files not found")
        return False
    
    # Load image and ground truth
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    gt_corners = np.array(annotation.get('corners', []))
    
    # Get predictions from all services
    from corner_detection_service import CornerDetectionService
    from optimized_corner_service import OptimizedCornerService
    
    cnn_service = CornerDetectionService()
    opt_service = OptimizedCornerService()
    yolo_service = YOLOCornerService(yolo_model_path)
    
    cnn_corners = cnn_service.detect_corners(image_path)
    opt_corners = opt_service.detect_corners(image_path)
    yolo_corners = yolo_service.detect_corners(image_path)
    
    # Create four-way comparison
    h, w = image.shape[:2]
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Ground truth (top-left)
    gt_image = image.copy()
    for i, corner in enumerate(gt_corners):
        cv2.circle(gt_image, tuple(corner.astype(int)), 25, (0, 255, 0), -1)
        cv2.putText(gt_image, f'GT{i}', tuple(corner.astype(int) + [-20, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(gt_image, "GROUND TRUTH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # CNN Original (top-right)
    cnn_image = image.copy()
    if cnn_corners:
        for i, corner in enumerate(cnn_corners):
            cv2.circle(cnn_image, tuple(np.array(corner).astype(int)), 25, (0, 0, 255), -1)
            cv2.putText(cnn_image, f'CNN{i}', tuple(np.array(corner).astype(int) + [-20, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cnn_error = np.mean(np.sqrt(np.sum((gt_corners - np.array(cnn_corners)) ** 2, axis=1)))
        cv2.putText(cnn_image, f"CNN ORIGINAL ({cnn_error:.1f}px)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(cnn_image, "CNN ORIGINAL (FAILED)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Optimized CNN (bottom-left)
    opt_image = image.copy()
    if opt_corners:
        for i, corner in enumerate(opt_corners):
            cv2.circle(opt_image, tuple(np.array(corner).astype(int)), 25, (255, 0, 0), -1)
            cv2.putText(opt_image, f'OPT{i}', tuple(np.array(corner).astype(int) + [-20, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        opt_error = np.mean(np.sqrt(np.sum((gt_corners - np.array(opt_corners)) ** 2, axis=1)))
        cv2.putText(opt_image, f"OPTIMIZED CNN ({opt_error:.1f}px)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    else:
        cv2.putText(opt_image, "OPTIMIZED CNN (FAILED)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # YOLO (bottom-right)
    yolo_image = image.copy()
    if yolo_corners:
        for i, corner in enumerate(yolo_corners):
            cv2.circle(yolo_image, tuple(np.array(corner).astype(int)), 25, (255, 255, 0), -1)
            cv2.putText(yolo_image, f'YOLO{i}', tuple(np.array(corner).astype(int) + [-20, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        yolo_error = np.mean(np.sqrt(np.sum((gt_corners - np.array(yolo_corners)) ** 2, axis=1)))
        cv2.putText(yolo_image, f"YOLO ({yolo_error:.1f}px)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    else:
        cv2.putText(yolo_image, "YOLO (TRAINING...)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # Assemble canvas
    canvas[:h, :w] = gt_image
    canvas[:h, w:] = cnn_image
    canvas[h:, :w] = opt_image
    canvas[h:, w:] = yolo_image
    
    # Add separators
    cv2.line(canvas, (w, 0), (w, h * 2), (255, 255, 255), 3)
    cv2.line(canvas, (0, h), (w * 2, h), (255, 255, 255), 3)
    
    # Save comparison
    output_path = "yolo_vs_cnn_comparison.jpg"
    cv2.imwrite(output_path, canvas)
    
    print(f"✅ Four-way comparison saved: {output_path}")
    return True

def main():
    """Main comparison function"""
    print("YOLO vs CNN Corner Detection Comparison")
    print("=" * 50)
    
    # Check YOLO training status
    yolo_ready, yolo_model_path = check_yolo_training_status()
    
    if yolo_ready:
        print("🎯 YOLO model is ready! Running full comparison...")
        
        # Run comprehensive comparison
        results, stats = compare_all_corner_detection_methods()
        
        # Create visualization
        create_yolo_corner_visualization()
        
        print(f"\n🎯 COMPARISON COMPLETE!")
        print("   Check the generated visualization files")
        
    else:
        print("⏳ YOLO model is still training...")
        print("   You can run this script again once training is complete")
        print("   Or check training progress in the background")
        
        # Create visualization framework for when YOLO is ready
        create_yolo_corner_visualization()
        
    print(f"\n💡 TO MONITOR YOLO TRAINING:")
    print("   Check the background process or logs")
    print("   Training typically takes 30-60 minutes")
    print("   Model will be saved to yolo_training_runs/")

if __name__ == "__main__":
    main()
