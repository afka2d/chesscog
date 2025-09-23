#!/usr/bin/env python3
"""
Fast Precision Corner Detector
===============================

Optimized corner detection that balances accuracy and speed.
Target: Under 3 seconds while improving on YOLO-only accuracy.

Strategy:
1. YOLO for robust initial detection (fast)
2. Single-pass sub-pixel refinement (fast)
3. Lightweight geometric validation (fast)
4. Optional edge verification only if time permits

Optimizations:
- Single preprocessing strategy (best performing)
- Early stopping when good results found
- Reduced search space
- Optimized parameters
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import time

# Import YOLO detector
try:
    from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastPrecisionDetector:
    """
    Fast precision corner detector optimized for speed while maintaining accuracy
    """
    
    def __init__(self):
        self.yolo_detector = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = ImprovedYOLOCornerDetector()
                logger.info("✅ YOLO detector loaded for fast precision")
            except Exception as e:
                logger.warning(f"⚠️  YOLO detector failed: {e}")
        
        # Optimized parameters for speed
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Reduced iterations
        self.subpix_winsize = (9, 9)  # Smaller window for speed
        
    def detect_corners_fast_precision(self, image_path: str, max_time: float = 3.0) -> Optional[List[List[float]]]:
        """
        Fast precision detection with time limit
        
        Args:
            image_path: Path to image
            max_time: Maximum processing time in seconds
            
        Returns:
            List of 4 corners or None
        """
        start_time = time.time()
        logger.info(f"🚀 Fast precision detection: {Path(image_path).name} (max {max_time}s)")
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Stage 1: YOLO detection (fast, ~0.15s)
        yolo_corners = self._fast_yolo_detection(image_path)
        if yolo_corners is None:
            logger.warning("YOLO failed, no fallback in fast mode")
            return None
        
        elapsed = time.time() - start_time
        remaining_time = max_time - elapsed
        logger.info(f"   YOLO: ✅ {elapsed:.3f}s, {remaining_time:.3f}s remaining")
        
        if remaining_time < 0.5:  # Not enough time for refinement
            return yolo_corners
        
        # Stage 2: Fast sub-pixel refinement (~0.1-0.5s)
        subpixel_corners = self._fast_subpixel_refinement(image, yolo_corners)
        working_corners = subpixel_corners if subpixel_corners else yolo_corners
        
        elapsed = time.time() - start_time
        remaining_time = max_time - elapsed
        logger.info(f"   Sub-pixel: ✅ {elapsed:.3f}s, {remaining_time:.3f}s remaining")
        
        if remaining_time < 0.5:  # Not enough time for more processing
            return working_corners
        
        # Stage 3: Fast geometric validation (~0.1-0.3s)
        geometric_corners = self._fast_geometric_validation(working_corners, image.shape)
        working_corners = geometric_corners if geometric_corners else working_corners
        
        elapsed = time.time() - start_time
        remaining_time = max_time - elapsed
        logger.info(f"   Geometric: ✅ {elapsed:.3f}s, {remaining_time:.3f}s remaining")
        
        # Stage 4: Optional edge refinement if time permits (skip if < 1s remaining)
        if remaining_time > 1.0:
            edge_corners = self._fast_edge_refinement(image, working_corners, max_time=remaining_time-0.1)
            working_corners = edge_corners if edge_corners else working_corners
            
            elapsed = time.time() - start_time
            logger.info(f"   Edge: ✅ {elapsed:.3f}s total")
        else:
            logger.info(f"   Edge: ⏭️  Skipped (insufficient time)")
        
        total_time = time.time() - start_time
        logger.info(f"🏆 Fast precision complete: {total_time:.3f}s total")
        
        return working_corners
    
    def _fast_yolo_detection(self, image_path: str) -> Optional[List[List[float]]]:
        """Fast YOLO detection"""
        if self.yolo_detector is None:
            return None
        
        try:
            corners = self.yolo_detector.detect_corners(image_path)
            if corners is not None:
                if isinstance(corners, np.ndarray):
                    corners = corners.tolist()
                return corners
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return None
    
    def _fast_subpixel_refinement(self, image: np.ndarray, corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Fast sub-pixel refinement with single strategy"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            # Single-pass sub-pixel refinement with optimized parameters
            refined_corners = cv2.cornerSubPix(gray, corners_np, self.subpix_winsize, (-1, -1), self.subpix_criteria)
            
            # Quick validation - ensure corners didn't move too much
            movements = np.linalg.norm(refined_corners - corners_np, axis=1)
            max_movement = np.max(movements)
            avg_movement = np.mean(movements)
            
            if max_movement < 30 and avg_movement < 15:  # Reasonable movement
                logger.info(f"     Sub-pixel: {avg_movement:.1f}px avg movement")
                return refined_corners.tolist()
            else:
                logger.warning(f"     Sub-pixel: Large movement {max_movement:.1f}px, keeping original")
                return corners
                
        except Exception as e:
            logger.warning(f"Fast sub-pixel failed: {e}")
            return None
    
    def _fast_geometric_validation(self, corners: List[List[float]], image_shape: Tuple[int, int, int]) -> Optional[List[List[float]]]:
        """Fast geometric validation and light correction"""
        try:
            corners_np = np.array(corners, dtype=np.float32)
            h, w = image_shape[:2]
            
            # Quick convexity check
            if not self._quick_convexity_check(corners_np):
                # Simple reordering attempt
                corners_np = self._quick_reorder_corners(corners_np)
            
            # Quick aspect ratio check
            width = np.linalg.norm(corners_np[1] - corners_np[0])
            height = np.linalg.norm(corners_np[3] - corners_np[0])
            aspect_ratio = width / height
            
            # Apply light correction if aspect ratio is very off
            if not (0.6 <= aspect_ratio <= 1.7):
                corrected_corners = self._quick_aspect_correction(corners_np, image_shape)
                if corrected_corners is not None:
                    logger.info("     Geometric: Light aspect correction applied")
                    return corrected_corners.tolist()
            
            # Ensure corners are within bounds
            corners_np[:, 0] = np.clip(corners_np[:, 0], 0, w-1)
            corners_np[:, 1] = np.clip(corners_np[:, 1], 0, h-1)
            
            logger.info("     Geometric: Validation passed")
            return corners_np.tolist()
            
        except Exception as e:
            logger.warning(f"Fast geometric validation failed: {e}")
            return None
    
    def _fast_edge_refinement(self, image: np.ndarray, corners: List[List[float]], max_time: float = 1.0) -> Optional[List[List[float]]]:
        """Fast edge-based refinement with time limit"""
        start_time = time.time()
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            # Create focused mask around corners
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corners_np.astype(np.int32)], 255)
            
            # Quick edge detection
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.bitwise_and(edges, mask)
            
            # Check time
            if time.time() - start_time > max_time * 0.5:
                return None
            
            # Fast line detection with reduced parameters
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
            
            if lines is not None and len(lines) >= 4:
                # Quick line grouping
                horizontal_lines, vertical_lines = self._quick_group_lines(lines)
                
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    # Select boundary lines quickly
                    boundary_lines = self._quick_select_boundary_lines(horizontal_lines, vertical_lines)
                    
                    if boundary_lines and time.time() - start_time < max_time:
                        # Calculate intersections
                        line_corners = self._quick_line_intersections(boundary_lines)
                        
                        if line_corners is not None:
                            # Quick validation
                            if self._quick_validate_line_corners(line_corners, corners_np):
                                elapsed = time.time() - start_time
                                logger.info(f"     Edge: Refined in {elapsed:.3f}s")
                                return line_corners.tolist()
            
        except Exception as e:
            logger.warning(f"Fast edge refinement failed: {e}")
        
        return None
    
    def _quick_convexity_check(self, corners: np.ndarray) -> bool:
        """Quick convexity check"""
        try:
            cross_products = []
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                p3 = corners[(i + 2) % 4]
                
                cross = np.cross(p2 - p1, p3 - p2)
                cross_products.append(cross)
            
            # Check if all have same sign
            signs = [np.sign(cp) for cp in cross_products if abs(cp) > 1e-6]
            return len(set(signs)) <= 1
            
        except:
            return False
    
    def _quick_reorder_corners(self, corners: np.ndarray) -> np.ndarray:
        """Quick corner reordering"""
        try:
            # Simple ordering by coordinate sums and differences
            sums = corners[:, 0] + corners[:, 1]
            diffs = corners[:, 0] - corners[:, 1]
            
            tl_idx = np.argmin(sums)  # Top-left: smallest sum
            br_idx = np.argmax(sums)  # Bottom-right: largest sum
            tr_idx = np.argmax(diffs)  # Top-right: largest x-y
            bl_idx = np.argmin(diffs)  # Bottom-left: smallest x-y
            
            # Validate unique indices
            indices = [tl_idx, tr_idx, br_idx, bl_idx]
            if len(set(indices)) == 4:
                return corners[indices]
            
            # Fallback: return original
            return corners
            
        except:
            return corners
    
    def _quick_aspect_correction(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Quick aspect ratio correction"""
        try:
            # Calculate center and average size
            center = np.mean(corners, axis=0)
            
            # Calculate side lengths
            side_lengths = []
            for i in range(4):
                side_length = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
                side_lengths.append(side_length)
            
            avg_side = np.mean(side_lengths)
            
            # Create more square-like corners
            half_size = avg_side / 2.2  # Slightly smaller for better fit
            
            corrected_corners = np.array([
                [center[0] - half_size, center[1] - half_size],  # TL
                [center[0] + half_size, center[1] - half_size],  # TR
                [center[0] + half_size, center[1] + half_size],  # BR
                [center[0] - half_size, center[1] + half_size]   # BL
            ])
            
            # Blend with original (light correction)
            weight = 0.2  # 20% correction
            final_corners = (1 - weight) * corners + weight * corrected_corners
            
            return final_corners
            
        except:
            return None
    
    def _quick_group_lines(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Quick line grouping by orientation"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines[:20]:  # Limit to first 20 lines for speed
            x1, y1, x2, y2 = line[0]
            
            # Quick angle calculation
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # Simple orientation test
            if dx > dy * 2:  # More horizontal
                horizontal_lines.append(line[0])
            elif dy > dx * 2:  # More vertical
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def _quick_select_boundary_lines(self, h_lines: List[np.ndarray], v_lines: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """Quick boundary line selection"""
        try:
            if len(h_lines) < 2 or len(v_lines) < 2:
                return None
            
            # Sort and select extremes
            h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
            v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)
            
            return [h_sorted[0], v_sorted[-1], h_sorted[-1], v_sorted[0]]  # top, right, bottom, left
            
        except:
            return None
    
    def _quick_line_intersections(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Quick line intersection calculation"""
        try:
            def intersect(line1, line2):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return np.array([x, y])
            
            top, right, bottom, left = lines
            
            tl = intersect(top, left)
            tr = intersect(top, right)
            br = intersect(bottom, right)
            bl = intersect(bottom, left)
            
            if all(c is not None for c in [tl, tr, br, bl]):
                return np.array([tl, tr, br, bl])
                
        except:
            pass
        
        return None
    
    def _quick_validate_line_corners(self, line_corners: np.ndarray, ref_corners: np.ndarray) -> bool:
        """Quick validation of line-fitted corners"""
        try:
            # Check that line corners are not too far from reference
            distances = np.linalg.norm(line_corners - ref_corners, axis=1)
            max_distance = np.max(distances)
            
            # Accept if improvement is reasonable
            return max_distance < 50  # pixels
            
        except:
            return False
    
    def evaluate_fast_precision(self, test_images_dir: str, annotations_dir: str, 
                               num_samples: int = 30, max_time: float = 3.0) -> Dict:
        """Evaluate fast precision detector"""
        logger.info(f"🧪 FAST PRECISION EVALUATION")
        logger.info("=" * 50)
        logger.info(f"Target: Under {max_time}s per image")
        
        test_images = list(Path(test_images_dir).glob("*.JPG"))[:num_samples]
        
        results = {
            'total_images': len(test_images),
            'successful_detections': 0,
            'errors': [],
            'processing_times': [],
            'time_budget_met': 0,
            'improvements_over_yolo': []
        }
        
        for img_path in test_images:
            try:
                # Load ground truth
                ann_path = Path(annotations_dir) / f"{img_path.stem}.json"
                if not ann_path.exists():
                    continue
                
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                
                gt_corners = annotation.get('corners', [])
                if len(gt_corners) != 4:
                    continue
                
                # Test YOLO-only for comparison
                yolo_corners = self._fast_yolo_detection(str(img_path))
                yolo_error = None
                if yolo_corners:
                    yolo_error = self._calculate_error(gt_corners, yolo_corners)
                
                # Test fast precision
                start_time = time.time()
                fast_corners = self.detect_corners_fast_precision(str(img_path), max_time)
                processing_time = time.time() - start_time
                
                if fast_corners is not None:
                    results['successful_detections'] += 1
                    
                    # Calculate error
                    fast_error = self._calculate_error(gt_corners, fast_corners)
                    results['errors'].append(fast_error)
                    results['processing_times'].append(processing_time)
                    
                    # Check if time budget was met
                    if processing_time <= max_time:
                        results['time_budget_met'] += 1
                    
                    # Compare with YOLO
                    if yolo_error is not None:
                        improvement = ((yolo_error - fast_error) / yolo_error) * 100
                        results['improvements_over_yolo'].append(improvement)
                        
                        status = "⚡" if processing_time <= max_time else "⏰"
                        logger.info(f"   {status} {img_path.name}: {fast_error:.1f}px ({processing_time:.3f}s) vs YOLO: {yolo_error:.1f}px ({improvement:+.1f}%)")
                    else:
                        status = "⚡" if processing_time <= max_time else "⏰"
                        logger.info(f"   {status} {img_path.name}: {fast_error:.1f}px ({processing_time:.3f}s)")
                
            except Exception as e:
                logger.warning(f"Error evaluating {img_path.name}: {e}")
        
        # Calculate summary
        if results['errors']:
            results['average_error'] = np.mean(results['errors'])
            results['median_error'] = np.median(results['errors'])
            results['std_error'] = np.std(results['errors'])
            results['max_error'] = np.max(results['errors'])
            results['min_error'] = np.min(results['errors'])
            results['success_rate'] = (results['successful_detections'] / results['total_images']) * 100
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['time_budget_success_rate'] = (results['time_budget_met'] / results['successful_detections']) * 100
            
            if results['improvements_over_yolo']:
                results['avg_improvement_over_yolo'] = np.mean(results['improvements_over_yolo'])
        
        self._print_fast_evaluation_summary(results, max_time)
        return results
    
    def _calculate_error(self, gt_corners: List[List[float]], detected_corners: List[List[float]]) -> float:
        """Calculate average corner error"""
        gt_np = np.array(gt_corners)
        det_np = np.array(detected_corners)
        
        errors = np.linalg.norm(gt_np - det_np, axis=1)
        return np.mean(errors)
    
    def _print_fast_evaluation_summary(self, results: Dict, max_time: float):
        """Print fast evaluation summary"""
        logger.info(f"\n🏆 FAST PRECISION DETECTOR RESULTS")
        logger.info("=" * 50)
        logger.info(f"📊 Images processed: {results['total_images']}")
        logger.info(f"✅ Successful detections: {results['successful_detections']}")
        logger.info(f"📈 Success rate: {results.get('success_rate', 0):.1f}%")
        
        if results['errors']:
            logger.info(f"\n🎯 ACCURACY METRICS:")
            logger.info(f"   Average error: {results['average_error']:.1f} pixels")
            logger.info(f"   Median error: {results['median_error']:.1f} pixels")
            logger.info(f"   Error range: {results['min_error']:.1f} - {results['max_error']:.1f} pixels")
            
            logger.info(f"\n⚡ SPEED METRICS:")
            logger.info(f"   Average time: {results['avg_processing_time']:.3f} seconds")
            logger.info(f"   Time budget ({max_time}s): {results['time_budget_success_rate']:.1f}% success")
            
            if results['improvements_over_yolo']:
                logger.info(f"\n🚀 VS YOLO-ONLY:")
                logger.info(f"   Average improvement: {results['avg_improvement_over_yolo']:+.1f}%")

def main():
    """Test the fast precision detector"""
    print("🚀 FAST PRECISION CORNER DETECTOR")
    print("=" * 50)
    print("Optimized for speed while maintaining accuracy improvement")
    print("Target: Under 3 seconds per image")
    print()
    
    detector = FastPrecisionDetector()
    
    # Test on sample image
    test_image = "my_chess_images/train/images/IMG_4698.JPG"
    if Path(test_image).exists():
        print(f"🧪 Testing on: {test_image}")
        
        # Test with different time budgets
        for time_budget in [1.0, 2.0, 3.0]:
            print(f"\n⏱️  Time budget: {time_budget}s")
            corners = detector.detect_corners_fast_precision(test_image, time_budget)
            
            if corners:
                print(f"✅ Success with {time_budget}s budget")
            else:
                print(f"❌ Failed with {time_budget}s budget")
    
    # Evaluate on test data
    test_dirs = [
        ("grey_background_dataset/images/test", "grey_background_dataset/annotations/test"),
        ("grey_background_dataset/images/val", "grey_background_dataset/annotations/val")
    ]
    
    for img_dir, ann_dir in test_dirs:
        if Path(img_dir).exists() and Path(ann_dir).exists():
            print(f"\n📊 Evaluating on {img_dir} (3s budget)...")
            results = detector.evaluate_fast_precision(img_dir, ann_dir, num_samples=15, max_time=3.0)
            
            if results:
                print(f"🎯 Results: {results['average_error']:.1f}px avg, {results['avg_processing_time']:.3f}s avg")
                print(f"⚡ Time budget success: {results['time_budget_success_rate']:.1f}%")

if __name__ == "__main__":
    main()
