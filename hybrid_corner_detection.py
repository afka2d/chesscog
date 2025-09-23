#!/usr/bin/env python3
"""
Hybrid Corner Detection System
==============================

Combines YOLO's robust detection with OpenCV's precision for maximum accuracy.

Pipeline:
1. YOLO: Robust initial detection (finds the general area)
2. OpenCV: Precise corner refinement using cv2.findChessboardCorners
3. Sub-pixel: Further refinement with cv2.cornerSubPix
4. Geometric: Apply geometric constraints and homography correction
5. Line fitting: Use edge detection + line intersection as fallback
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json
import time

# Import our existing YOLO detector
try:
    from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLO detector not available")
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridCornerDetector:
    """
    Hybrid corner detection combining YOLO + OpenCV for maximum precision
    """
    
    def __init__(self):
        self.yolo_detector = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = ImprovedYOLOCornerDetector()
                logger.info("✅ YOLO detector loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  YOLO detector failed to load: {e}")
        
        # Chessboard detection parameters
        self.chessboard_size = (7, 7)  # Internal corners (8x8 board has 7x7 internal corners)
        
    def detect_corners(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Main corner detection pipeline
        
        Returns:
            List of 4 corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None
        """
        logger.info(f"🎯 Starting hybrid corner detection for {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Stage 1: YOLO initial detection
        yolo_corners = self._yolo_detection(image_path)
        if yolo_corners is None:
            logger.warning("YOLO detection failed, trying OpenCV-only approach")
            return self._opencv_only_detection(image)
        
        logger.info(f"🎯 YOLO detected corners: {yolo_corners}")
        
        # Stage 2: OpenCV chessboard refinement
        refined_corners = self._opencv_chessboard_refinement(image, yolo_corners)
        if refined_corners is not None:
            logger.info("✅ OpenCV chessboard refinement successful")
            corners = refined_corners
        else:
            logger.info("⚠️  OpenCV chessboard refinement failed, using YOLO corners")
            corners = yolo_corners
        
        # Stage 3: Sub-pixel refinement
        subpixel_corners = self._subpixel_refinement(image, corners)
        if subpixel_corners is not None:
            logger.info("✅ Sub-pixel refinement successful")
            corners = subpixel_corners
        
        # Stage 4: Geometric constraints
        geometric_corners = self._apply_geometric_constraints(corners, image.shape)
        if geometric_corners is not None:
            logger.info("✅ Geometric constraints applied")
            corners = geometric_corners
        
        # Stage 5: Line fitting refinement (fallback/verification)
        line_corners = self._line_fitting_refinement(image, corners)
        if line_corners is not None:
            logger.info("✅ Line fitting refinement successful")
            corners = line_corners
        
        logger.info(f"🏆 Final refined corners: {corners}")
        return corners
    
    def _yolo_detection(self, image_path: str) -> Optional[List[List[float]]]:
        """Stage 1: YOLO initial detection"""
        if self.yolo_detector is None:
            return None
        
        try:
            corners = self.yolo_detector.detect_corners(image_path)
            if corners is not None:
                # Convert to proper format
                if isinstance(corners, np.ndarray):
                    corners = corners.tolist()
                return corners
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return None
    
    def _opencv_chessboard_refinement(self, image: np.ndarray, initial_corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 2: OpenCV chessboard pattern refinement"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create region of interest around YOLO's detection
            corners_np = np.array(initial_corners, dtype=np.float32)
            
            # Expand the region slightly for chessboard detection
            margin = 100
            x_min = max(0, int(np.min(corners_np[:, 0]) - margin))
            y_min = max(0, int(np.min(corners_np[:, 1]) - margin))
            x_max = min(gray.shape[1], int(np.max(corners_np[:, 0]) + margin))
            y_max = min(gray.shape[0], int(np.max(corners_np[:, 1]) + margin))
            
            # Extract ROI
            roi = gray[y_min:y_max, x_min:x_max]
            
            # Try to find chessboard corners in ROI
            found, corners_roi = cv2.findChessboardCorners(
                roi, 
                self.chessboard_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            if found and corners_roi is not None:
                logger.info(f"🎯 Found {len(corners_roi)} internal chessboard corners")
                
                # Convert back to full image coordinates
                corners_roi[:, :, 0] += x_min
                corners_roi[:, :, 1] += y_min
                
                # Extract the 4 outer corners from the 7x7 grid
                board_corners = self._extract_board_corners_from_grid(corners_roi)
                
                if board_corners is not None:
                    return board_corners.tolist()
            
        except Exception as e:
            logger.warning(f"OpenCV chessboard refinement failed: {e}")
        
        return None
    
    def _extract_board_corners_from_grid(self, internal_corners: np.ndarray) -> Optional[np.ndarray]:
        """Extract the 4 board corners from 7x7 internal corner grid"""
        try:
            # Reshape to 7x7 grid
            corners_grid = internal_corners.reshape(7, 7, 2)
            
            # The board corners are extrapolated from the internal corners
            # Top-left: extrapolate from (0,0) corner
            tl_internal = corners_grid[0, 0]
            tr_internal = corners_grid[0, 6]
            bl_internal = corners_grid[6, 0]
            br_internal = corners_grid[6, 6]
            
            # Calculate the square size to extrapolate outward
            square_width = (tr_internal[0] - tl_internal[0]) / 6
            square_height = (bl_internal[1] - tl_internal[1]) / 6
            
            # Extrapolate to actual board corners
            tl_board = tl_internal - [square_width, square_height]
            tr_board = tr_internal + [square_width, -square_height]
            br_board = br_internal + [square_width, square_height]
            bl_board = bl_internal - [square_width, -square_height]
            
            board_corners = np.array([tl_board, tr_board, br_board, bl_board])
            
            return board_corners
            
        except Exception as e:
            logger.warning(f"Could not extract board corners from grid: {e}")
            return None
    
    def _opencv_only_detection(self, image: np.ndarray) -> Optional[List[List[float]]]:
        """Fallback: OpenCV-only detection when YOLO fails"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try to find chessboard corners in full image
            found, corners = cv2.findChessboardCorners(
                gray, 
                self.chessboard_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if found and corners is not None:
                logger.info("✅ OpenCV-only detection successful")
                board_corners = self._extract_board_corners_from_grid(corners)
                if board_corners is not None:
                    return board_corners.tolist()
            
        except Exception as e:
            logger.warning(f"OpenCV-only detection failed: {e}")
        
        return None
    
    def _subpixel_refinement(self, image: np.ndarray, corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 3: Sub-pixel corner refinement"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            # Sub-pixel refinement parameters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            winSize = (11, 11)
            zeroZone = (-1, -1)
            
            # Refine corners
            refined_corners = cv2.cornerSubPix(gray, corners_np, winSize, zeroZone, criteria)
            
            # Check if refinement was successful (corners didn't move too much)
            max_movement = np.max(np.linalg.norm(refined_corners - corners_np, axis=1))
            if max_movement < 50:  # Reasonable movement threshold
                return refined_corners.tolist()
            else:
                logger.warning(f"Sub-pixel refinement moved corners too much: {max_movement:.1f} pixels")
                
        except Exception as e:
            logger.warning(f"Sub-pixel refinement failed: {e}")
        
        return None
    
    def _apply_geometric_constraints(self, corners: List[List[float]], image_shape: Tuple[int, int, int]) -> Optional[List[List[float]]]:
        """Stage 4: Apply geometric constraints"""
        try:
            corners_np = np.array(corners, dtype=np.float32)
            
            # Check if corners form a valid convex quadrilateral
            if not self._is_convex_quadrilateral(corners_np):
                logger.warning("Corners do not form a convex quadrilateral")
                # Try to fix by reordering
                corners_np = self._reorder_corners(corners_np)
            
            # Apply homography correction to make it more rectangular
            corrected_corners = self._homography_correction(corners_np, image_shape)
            
            if corrected_corners is not None:
                return corrected_corners.tolist()
                
        except Exception as e:
            logger.warning(f"Geometric constraints failed: {e}")
        
        return None
    
    def _line_fitting_refinement(self, image: np.ndarray, corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 5: Line fitting refinement for ultimate precision"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            # Create a mask for the chessboard region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [corners_np.astype(np.int32)], 255)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.bitwise_and(edges, mask)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) >= 4:
                # Group lines by orientation (horizontal vs vertical)
                horizontal_lines, vertical_lines = self._group_lines_by_orientation(lines)
                
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    # Find the best 4 lines (2 horizontal, 2 vertical)
                    best_lines = self._select_best_board_lines(horizontal_lines, vertical_lines, corners_np)
                    
                    # Calculate intersections
                    line_corners = self._calculate_line_intersections(best_lines)
                    
                    if line_corners is not None:
                        logger.info("✅ Line fitting refinement successful")
                        return line_corners.tolist()
            
        except Exception as e:
            logger.warning(f"Line fitting refinement failed: {e}")
        
        return None
    
    def _is_convex_quadrilateral(self, corners: np.ndarray) -> bool:
        """Check if 4 corners form a convex quadrilateral"""
        try:
            # Reorder corners to ensure proper winding
            ordered_corners = self._reorder_corners(corners)
            
            # Check if all cross products have the same sign (convex test)
            for i in range(4):
                p1 = ordered_corners[i]
                p2 = ordered_corners[(i + 1) % 4]
                p3 = ordered_corners[(i + 2) % 4]
                
                cross = np.cross(p2 - p1, p3 - p2)
                if i == 0:
                    sign = np.sign(cross)
                elif np.sign(cross) != sign:
                    return False
            
            return True
            
        except:
            return False
    
    def _reorder_corners(self, corners: np.ndarray) -> np.ndarray:
        """Reorder corners to top-left, top-right, bottom-right, bottom-left"""
        # Find center point
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder starting from top-left (smallest angle)
        ordered_corners = corners[sorted_indices]
        
        return ordered_corners
    
    def _homography_correction(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Apply homography correction to make the quadrilateral more rectangular"""
        try:
            h, w = image_shape[:2]
            
            # Define ideal rectangle proportions (slightly rectangular, like a real chessboard)
            aspect_ratio = 1.0  # Square chessboard
            ideal_size = min(w, h) * 0.8  # 80% of image size
            
            # Define target rectangle corners
            target_corners = np.array([
                [w/2 - ideal_size/2, h/2 - ideal_size/2],  # Top-left
                [w/2 + ideal_size/2, h/2 - ideal_size/2],  # Top-right  
                [w/2 + ideal_size/2, h/2 + ideal_size/2],  # Bottom-right
                [w/2 - ideal_size/2, h/2 + ideal_size/2]   # Bottom-left
            ], dtype=np.float32)
            
            # Calculate homography
            homography, _ = cv2.findHomography(corners, target_corners, cv2.RANSAC)
            
            if homography is not None:
                # Apply inverse homography to get corrected corners
                corrected_corners = cv2.perspectiveTransform(
                    target_corners.reshape(-1, 1, 2), 
                    np.linalg.inv(homography)
                ).reshape(-1, 2)
                
                return corrected_corners
                
        except Exception as e:
            logger.warning(f"Homography correction failed: {e}")
        
        return None
    
    def _group_lines_by_orientation(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Group lines into horizontal and vertical based on angle"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Normalize angle to [0, 180)
            angle = abs(angle)
            if angle > 90:
                angle = 180 - angle
            
            # Classify as horizontal or vertical
            if angle < 30:  # More horizontal
                horizontal_lines.append(line[0])
            elif angle > 60:  # More vertical
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def _select_best_board_lines(self, horizontal_lines: List[np.ndarray], vertical_lines: List[np.ndarray], 
                                corners_ref: np.ndarray) -> Optional[List[np.ndarray]]:
        """Select the 4 best lines that represent the chessboard edges"""
        try:
            # Sort horizontal lines by y-coordinate (top and bottom)
            h_lines_sorted = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2)
            top_line = h_lines_sorted[0] if h_lines_sorted else None
            bottom_line = h_lines_sorted[-1] if h_lines_sorted else None
            
            # Sort vertical lines by x-coordinate (left and right)
            v_lines_sorted = sorted(vertical_lines, key=lambda line: (line[0] + line[2]) / 2)
            left_line = v_lines_sorted[0] if v_lines_sorted else None
            right_line = v_lines_sorted[-1] if v_lines_sorted else None
            
            if all(line is not None for line in [top_line, bottom_line, left_line, right_line]):
                return [top_line, right_line, bottom_line, left_line]
                
        except Exception as e:
            logger.warning(f"Line selection failed: {e}")
        
        return None
    
    def _calculate_line_intersections(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate intersections of 4 lines to get precise corners"""
        try:
            top_line, right_line, bottom_line, left_line = lines
            
            def line_intersection(line1, line2):
                """Calculate intersection of two lines"""
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                
                intersection_x = x1 + t * (x2 - x1)
                intersection_y = y1 + t * (y2 - y1)
                
                return np.array([intersection_x, intersection_y])
            
            # Calculate 4 corner intersections
            tl_corner = line_intersection(top_line, left_line)    # Top-left
            tr_corner = line_intersection(top_line, right_line)   # Top-right
            br_corner = line_intersection(bottom_line, right_line) # Bottom-right
            bl_corner = line_intersection(bottom_line, left_line)  # Bottom-left
            
            if all(corner is not None for corner in [tl_corner, tr_corner, br_corner, bl_corner]):
                corners = np.array([tl_corner, tr_corner, br_corner, bl_corner])
                return corners
                
        except Exception as e:
            logger.warning(f"Line intersection calculation failed: {e}")
        
        return None
    
    def evaluate_accuracy(self, test_images_dir: str, annotations_dir: str, num_samples: int = 20) -> dict:
        """Evaluate the hybrid detector against ground truth"""
        logger.info(f"🧪 EVALUATING HYBRID DETECTOR ACCURACY")
        logger.info("=" * 60)
        
        test_images = list(Path(test_images_dir).glob("*.JPG"))[:num_samples]
        
        errors = []
        successful_detections = 0
        
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
                
                # Detect corners with hybrid method
                start_time = time.time()
                detected_corners = self.detect_corners(str(img_path))
                detection_time = time.time() - start_time
                
                if detected_corners is not None:
                    successful_detections += 1
                    
                    # Calculate error
                    gt_np = np.array(gt_corners)
                    det_np = np.array(detected_corners)
                    
                    # Calculate per-corner error
                    corner_errors = np.linalg.norm(gt_np - det_np, axis=1)
                    avg_error = np.mean(corner_errors)
                    max_error = np.max(corner_errors)
                    
                    errors.append(avg_error)
                    
                    logger.info(f"   {img_path.name}: {avg_error:.1f}px avg, {max_error:.1f}px max ({detection_time:.3f}s)")
                
            except Exception as e:
                logger.warning(f"   Error evaluating {img_path.name}: {e}")
        
        if errors:
            results = {
                'num_images': len(test_images),
                'successful_detections': successful_detections,
                'success_rate': (successful_detections / len(test_images)) * 100,
                'average_error': np.mean(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors),
                'std_error': np.std(errors)
            }
            
            logger.info(f"\n🏆 HYBRID DETECTOR RESULTS:")
            logger.info(f"   Success rate: {results['success_rate']:.1f}%")
            logger.info(f"   Average error: {results['average_error']:.1f} pixels")
            logger.info(f"   Median error: {results['median_error']:.1f} pixels")
            logger.info(f"   Error range: {results['min_error']:.1f} - {results['max_error']:.1f} pixels")
            
            return results
        
        return {}

def main():
    """Test the hybrid corner detector"""
    print("🚀 HYBRID CORNER DETECTION SYSTEM")
    print("=" * 50)
    print("Combining YOLO robustness with OpenCV precision")
    print()
    
    detector = HybridCornerDetector()
    
    # Test on a sample image
    test_image = "my_chess_images/train/images/IMG_4698.JPG"
    if Path(test_image).exists():
        print(f"🧪 Testing on: {test_image}")
        corners = detector.detect_corners(test_image)
        
        if corners:
            print(f"✅ Detection successful!")
            print(f"   Corners: {corners}")
        else:
            print("❌ Detection failed")
    
    # Evaluate on available test data
    test_dirs = [
        ("grey_background_dataset/images/test", "grey_background_dataset/annotations/test"),
        ("grey_background_dataset/images/val", "grey_background_dataset/annotations/val")
    ]
    
    for img_dir, ann_dir in test_dirs:
        if Path(img_dir).exists() and Path(ann_dir).exists():
            print(f"\n📊 Evaluating on {img_dir}...")
            results = detector.evaluate_accuracy(img_dir, ann_dir, num_samples=10)
            
            if results:
                print(f"🎯 Results: {results['average_error']:.1f}px average error")

if __name__ == "__main__":
    main()
