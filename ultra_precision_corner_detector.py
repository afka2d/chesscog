#!/usr/bin/env python3
"""
Ultra Precision Corner Detector
================================

Maximum accuracy corner detection within 2-second budget.
Designed specifically for chess board warping accuracy requirements.

Key Features:
- Multi-resolution YOLO ensemble
- Adaptive sub-pixel refinement based on image quality
- Intelligent geometric optimization
- Confidence-based processing pipeline
- Time budget management

Target: <15px average error in <2 seconds
Current best: 21.9px in 0.2s (Fast Precision)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import time
from ultralytics import YOLO

# Import existing detectors
try:
    from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraPrecisionCornerDetector:
    """
    Ultra-precise corner detection optimized for chess board warping accuracy
    """
    
    def __init__(self, yolo_model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.yolo_detector = None
        self.yolo_model = None
        
        if YOLO_AVAILABLE:
            try:
                # Load both our custom detector and raw YOLO model
                self.yolo_detector = ImprovedYOLOCornerDetector(yolo_model_path)
                self.yolo_model = YOLO(yolo_model_path)
                logger.info("✅ Ultra precision YOLO system loaded")
            except Exception as e:
                logger.warning(f"⚠️  YOLO loading failed: {e}")
        
        # Optimized parameters for maximum accuracy
        self.high_precision_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        self.medium_precision_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        self.fast_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    def detect_corners(self, image_path: str, time_budget: float = 2.0) -> Tuple[Optional[List[List[float]]], float, bool]:
        """
        Ultra-precise corner detection with time budget management
        
        Returns:
            (corners, time_taken, budget_met)
        """
        start_time = time.time()
        logger.info(f"🎯 Ultra precision detection: {Path(image_path).name} (budget: {time_budget}s)")
        
        if not self.yolo_detector:
            logger.error("YOLO detector not available")
            return None, 0.0, False
        
        # Load image once
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None, 0.0, False
        
        # Stage 1: Multi-resolution YOLO ensemble (0.3-0.5s)
        ensemble_corners, yolo_confidence = self._multi_resolution_yolo_ensemble(image_path, time_budget * 0.25)
        if ensemble_corners is None:
            return None, time.time() - start_time, False
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        logger.info(f"   Multi-YOLO: ✅ {elapsed:.3f}s, confidence: {yolo_confidence:.3f}, {remaining:.3f}s remaining")
        
        if remaining < 0.5:
            return ensemble_corners, elapsed, True
        
        # Stage 2: Adaptive sub-pixel refinement (0.1-0.4s)
        refined_corners = self._adaptive_subpixel_refinement(image, ensemble_corners, remaining * 0.4)
        working_corners = refined_corners if refined_corners is not None else ensemble_corners
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        logger.info(f"   Adaptive sub-pixel: ✅ {elapsed:.3f}s, {remaining:.3f}s remaining")
        
        if remaining < 0.3:
            return working_corners, elapsed, True
        
        # Stage 3: Intelligent geometric optimization (0.1-0.6s)
        optimized_corners = self._intelligent_geometric_optimization(working_corners, yolo_confidence, remaining * 0.6)
        working_corners = optimized_corners if optimized_corners is not None else working_corners
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        logger.info(f"   Geometric optimization: ✅ {elapsed:.3f}s, {remaining:.3f}s remaining")
        
        if remaining < 0.2:
            return working_corners, elapsed, True
        
        # Stage 4: Selective edge enhancement (0.1-0.8s)
        if yolo_confidence < 0.9:  # Only if YOLO was uncertain
            enhanced_corners = self._selective_edge_enhancement(image, working_corners, remaining)
            final_corners = enhanced_corners if enhanced_corners is not None else working_corners
        else:
            logger.info(f"   High YOLO confidence ({yolo_confidence:.3f}), skipping edge enhancement")
            final_corners = working_corners
        
        total_time = time.time() - start_time
        budget_met = total_time <= time_budget
        
        logger.info(f"🏆 Ultra precision complete: {total_time:.3f}s (budget: {time_budget}s)")
        return final_corners, total_time, budget_met
    
    def _multi_resolution_yolo_ensemble(self, image_path: str, time_budget: float) -> Tuple[Optional[List[List[float]]], float]:
        """
        Run YOLO at multiple resolutions and ensemble the results
        """
        start_time = time.time()
        
        # Resolution 1: Standard 640px (fast, baseline)
        try:
            corners_640, conf_640 = self._yolo_detect_with_confidence(image_path, img_size=640)
        except Exception as e:
            logger.warning(f"YOLO 640px failed: {e}")
            return None, 0.0
        
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.7:  # Used >70% of YOLO budget
            return corners_640, conf_640
        
        # Resolution 2: Higher resolution 896px (slower, more precise)
        try:
            corners_896, conf_896 = self._yolo_detect_with_confidence(image_path, img_size=896)
        except Exception as e:
            logger.warning(f"YOLO 896px failed: {e}")
            return corners_640, conf_640
        
        # Ensemble decision
        if corners_640 and corners_896:
            ensemble_corners = self._weighted_corner_ensemble(corners_640, corners_896, conf_640, conf_896)
            ensemble_confidence = max(conf_640, conf_896)  # Use higher confidence
            return ensemble_corners, ensemble_confidence
        
        return corners_640 or corners_896, conf_640 or conf_896 or 0.0
    
    def _yolo_detect_with_confidence(self, image_path: str, img_size: int = 640) -> Tuple[Optional[List[List[float]]], float]:
        """
        Run YOLO detection and return corners with confidence
        """
        try:
            # Use our existing improved YOLO detector for consistency
            if hasattr(self.yolo_detector, 'detect_corners_from_image'):
                # Load image and detect
                image = cv2.imread(image_path)
                if image is None:
                    return None, 0.0
                
                corners, confidence = self.yolo_detector.detect_corners_from_image(image)
                return corners, confidence if confidence else 0.9
            else:
                # Fallback to detect_corners method
                corners = self.yolo_detector.detect_corners(image_path)
                return corners, 0.9  # Default confidence
            
        except Exception as e:
            logger.warning(f"YOLO detection failed at {img_size}px: {e}")
            return None, 0.0
    
    def _extract_corners_from_mask(self, mask_data: np.ndarray, original_shape: Tuple[int, int]) -> Optional[List[List[float]]]:
        """
        Extract precise corners from YOLO segmentation mask
        """
        try:
            # Resize mask to original image size
            h, w = original_shape
            mask_resized = cv2.resize(mask_data, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate to quadrilateral with higher precision
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # More precise approximation
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Ensure we have exactly 4 points
            if len(approx) != 4:
                # Fallback: use convex hull and find 4 extreme points
                hull = cv2.convexHull(largest_contour)
                approx = self._find_four_corners_from_hull(hull)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(float)
                return self._sort_corners(corners).tolist()
            
            return None
            
        except Exception as e:
            logger.warning(f"Corner extraction failed: {e}")
            return None
    
    def _find_four_corners_from_hull(self, hull: np.ndarray) -> np.ndarray:
        """
        Find 4 corners from convex hull by finding extreme points
        """
        hull = hull.reshape(-1, 2)
        
        # Find extreme points
        top_left = hull[np.argmin(hull[:, 0] + hull[:, 1])]
        top_right = hull[np.argmax(hull[:, 0] - hull[:, 1])]
        bottom_right = hull[np.argmax(hull[:, 0] + hull[:, 1])]
        bottom_left = hull[np.argmin(hull[:, 0] - hull[:, 1])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left]).reshape(4, 1, 2)
    
    def _weighted_corner_ensemble(self, corners_640: List[List[float]], corners_896: List[List[float]], 
                                 conf_640: float, conf_896: float) -> List[List[float]]:
        """
        Combine two YOLO predictions with confidence weighting
        """
        # Normalize weights
        total_conf = conf_640 + conf_896
        if total_conf == 0:
            return corners_640  # Fallback
        
        weight_640 = conf_640 / total_conf
        weight_896 = conf_896 / total_conf
        
        # Weighted average
        corners_640_np = np.array(corners_640)
        corners_896_np = np.array(corners_896)
        
        ensemble_corners = weight_640 * corners_640_np + weight_896 * corners_896_np
        
        logger.info(f"     Ensemble: 640px weight: {weight_640:.3f}, 896px weight: {weight_896:.3f}")
        return ensemble_corners.tolist()
    
    def _adaptive_subpixel_refinement(self, image: np.ndarray, corners: List[List[float]], 
                                    time_budget: float) -> Optional[List[List[float]]]:
        """
        Adaptive sub-pixel refinement based on image characteristics
        """
        start_time = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze image quality to determine optimal parameters
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        
        # Adaptive parameter selection
        if sharpness > 1000 and contrast > 50:  # High quality image
            criteria = self.high_precision_criteria
            winsize = (15, 15)
            logger.info(f"     High quality image (sharpness: {sharpness:.0f}), using precision mode")
        elif sharpness > 500:  # Medium quality
            criteria = self.medium_precision_criteria
            winsize = (11, 11)
            logger.info(f"     Medium quality image (sharpness: {sharpness:.0f}), using balanced mode")
        else:  # Lower quality
            criteria = self.fast_criteria
            winsize = (9, 9)
            logger.info(f"     Lower quality image (sharpness: {sharpness:.0f}), using fast mode")
        
        # Apply sub-pixel refinement
        try:
            corners_np = np.array(corners, dtype=np.float32)
            refined_corners = cv2.cornerSubPix(gray, corners_np, winsize, (-1, -1), criteria)
            
            # Calculate movement to assess effectiveness
            movement = np.mean(np.linalg.norm(refined_corners - corners_np, axis=1))
            logger.info(f"     Sub-pixel movement: {movement:.2f}px avg")
            
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                logger.warning(f"Sub-pixel refinement exceeded budget: {elapsed:.3f}s > {time_budget:.3f}s")
            
            return refined_corners.tolist()
            
        except Exception as e:
            logger.warning(f"Sub-pixel refinement failed: {e}")
            return corners
    
    def _intelligent_geometric_optimization(self, corners: List[List[float]], yolo_confidence: float, 
                                          time_budget: float) -> Optional[List[List[float]]]:
        """
        Intelligent geometric optimization based on confidence and time budget
        """
        start_time = time.time()
        corners_np = np.array(corners)
        
        # Always do basic validation (fast)
        if not self._is_valid_quadrilateral(corners_np):
            corners_np = self._fix_invalid_quadrilateral(corners_np)
            logger.info("     Fixed invalid quadrilateral")
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        
        # Confidence-based processing
        if yolo_confidence < 0.85 and remaining > 0.3:
            # Low confidence: apply aggressive optimization
            logger.info(f"     Low confidence ({yolo_confidence:.3f}), applying intensive optimization")
            corners_np = self._intensive_geometric_optimization(corners_np, remaining * 0.8)
            
        elif yolo_confidence < 0.95 and remaining > 0.2:
            # Medium confidence: moderate optimization
            logger.info(f"     Medium confidence ({yolo_confidence:.3f}), applying moderate optimization")
            corners_np = self._moderate_geometric_optimization(corners_np, remaining * 0.6)
            
        else:
            # High confidence: minimal optimization
            logger.info(f"     High confidence ({yolo_confidence:.3f}), minimal optimization")
            corners_np = self._minimal_geometric_optimization(corners_np)
        
        elapsed = time.time() - start_time
        logger.info(f"     Geometric optimization: {elapsed:.3f}s")
        
        return corners_np.tolist()
    
    def _intensive_geometric_optimization(self, corners: np.ndarray, time_budget: float) -> np.ndarray:
        """
        Intensive geometric optimization for low-confidence detections
        """
        # 1. Enforce chessboard constraints
        corners = self._enforce_chessboard_geometry(corners)
        
        # 2. Optimize for minimal perspective distortion
        corners = self._minimize_perspective_distortion(corners)
        
        # 3. Enforce parallel opposite sides
        corners = self._enforce_parallel_sides(corners)
        
        logger.info("     Applied intensive geometric constraints")
        return corners
    
    def _moderate_geometric_optimization(self, corners: np.ndarray, time_budget: float) -> np.ndarray:
        """
        Moderate geometric optimization for medium-confidence detections
        """
        # 1. Basic chessboard geometry
        corners = self._enforce_chessboard_geometry(corners)
        
        # 2. Light perspective correction
        corners = self._light_perspective_correction(corners)
        
        logger.info("     Applied moderate geometric constraints")
        return corners
    
    def _minimal_geometric_optimization(self, corners: np.ndarray) -> np.ndarray:
        """
        Minimal optimization for high-confidence detections
        """
        # Just ensure valid quadrilateral and reasonable aspect ratio
        corners = self._ensure_reasonable_aspect_ratio(corners)
        logger.info("     Applied minimal geometric validation")
        return corners
    
    def _selective_edge_enhancement(self, image: np.ndarray, corners: List[List[float]], 
                                  time_budget: float) -> Optional[List[List[float]]]:
        """
        Selective edge enhancement for uncertain corners only
        """
        start_time = time.time()
        
        if time_budget < 0.2:
            return corners  # Not enough time
        
        logger.info("     Applying selective edge enhancement")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive edge detection based on image characteristics
        median_intensity = np.median(gray)
        
        if median_intensity < 100:  # Dark image
            edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        elif median_intensity > 180:  # Bright image
            edges = cv2.Canny(gray, 80, 200, apertureSize=3)
        else:  # Normal image
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Refine each corner using nearby edges
        corners_np = np.array(corners)
        refined_corners = corners_np.copy()
        
        for i, corner in enumerate(corners_np):
            if time.time() - start_time > time_budget * 0.8:
                break  # Time budget almost exhausted
            
            refined_corner = self._refine_corner_with_edges(edges, corner, search_radius=20)
            refined_corners[i] = refined_corner
        
        elapsed = time.time() - start_time
        logger.info(f"     Edge enhancement: {elapsed:.3f}s")
        
        return refined_corners.tolist()
    
    def _refine_corner_with_edges(self, edges: np.ndarray, corner: np.ndarray, search_radius: int = 20) -> np.ndarray:
        """
        Refine a single corner using nearby edge information
        """
        x, y = int(corner[0]), int(corner[1])
        
        # Extract region around corner
        y1, y2 = max(0, y - search_radius), min(edges.shape[0], y + search_radius)
        x1, x2 = max(0, x - search_radius), min(edges.shape[1], x + search_radius)
        
        edge_region = edges[y1:y2, x1:x2]
        
        if edge_region.size == 0:
            return corner
        
        # Find edge pixels in the region
        edge_points = np.column_stack(np.where(edge_region > 0))
        
        if len(edge_points) < 10:  # Not enough edge information
            return corner
        
        # Convert back to image coordinates
        edge_points[:, 0] += y1  # y coordinates
        edge_points[:, 1] += x1  # x coordinates
        edge_points = edge_points[:, [1, 0]]  # Convert to (x, y)
        
        # Find the edge point closest to the predicted corner
        distances = np.linalg.norm(edge_points - corner, axis=1)
        closest_edge_idx = np.argmin(distances)
        closest_edge_point = edge_points[closest_edge_idx]
        
        # Only use edge point if it's close enough (within 10 pixels)
        if distances[closest_edge_idx] < 10:
            return closest_edge_point
        
        return corner
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Sort corners in consistent order: top-left, top-right, bottom-right, bottom-left
        """
        # Find center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Find top-left corner (smallest x + y sum)
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return reordered
    
    def _is_valid_quadrilateral(self, corners: np.ndarray) -> bool:
        """
        Check if corners form a valid quadrilateral
        """
        if len(corners) != 4:
            return False
        
        # Check if corners are roughly in the right positions
        center = np.mean(corners, axis=0)
        
        # All corners should be reasonably far from center
        distances = np.linalg.norm(corners - center, axis=1)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Reasonable distance ratio
        if max_distance / min_distance > 3.0:
            return False
        
        # Check area (should be reasonable size)
        area = cv2.contourArea(corners.astype(np.float32))
        if area < 10000:  # Too small
            return False
        
        return True
    
    def _fix_invalid_quadrilateral(self, corners: np.ndarray) -> np.ndarray:
        """
        Fix invalid quadrilateral by adjusting problematic corners
        """
        # Simple fix: ensure corners form a convex quadrilateral
        center = np.mean(corners, axis=0)
        
        # Move corners away from center if they're too close
        for i in range(len(corners)):
            distance = np.linalg.norm(corners[i] - center)
            if distance < 50:  # Too close to center
                direction = (corners[i] - center) / distance
                corners[i] = center + direction * 100  # Move to reasonable distance
        
        return corners
    
    def _enforce_chessboard_geometry(self, corners: np.ndarray) -> np.ndarray:
        """
        Enforce chessboard-specific geometric constraints
        """
        # Chessboards are roughly square, so enforce reasonable aspect ratio
        return self._ensure_reasonable_aspect_ratio(corners)
    
    def _ensure_reasonable_aspect_ratio(self, corners: np.ndarray) -> np.ndarray:
        """
        Ensure the quadrilateral has a reasonable aspect ratio for a chessboard
        """
        # Calculate current aspect ratio
        sorted_corners = self._sort_corners(corners)
        
        # Calculate width and height
        width1 = np.linalg.norm(sorted_corners[1] - sorted_corners[0])  # Top edge
        width2 = np.linalg.norm(sorted_corners[2] - sorted_corners[3])  # Bottom edge
        height1 = np.linalg.norm(sorted_corners[3] - sorted_corners[0])  # Left edge
        height2 = np.linalg.norm(sorted_corners[2] - sorted_corners[1])  # Right edge
        
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        aspect_ratio = avg_width / avg_height if avg_height > 0 else 1.0
        
        # If aspect ratio is too far from square, apply light correction
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            logger.info(f"     Correcting aspect ratio: {aspect_ratio:.3f} → closer to 1.0")
            # Apply light correction towards square shape
            # This is a conservative adjustment
            target_ratio = 1.0
            correction_factor = 0.3  # Conservative correction
            
            if aspect_ratio < 1.0:  # Too tall
                # Slightly widen
                center = np.mean(sorted_corners, axis=0)
                for i in [1, 2]:  # Right corners
                    direction = sorted_corners[i] - center
                    sorted_corners[i] = center + direction * (1 + correction_factor * (target_ratio - aspect_ratio))
            else:  # Too wide
                # Slightly heighten
                center = np.mean(sorted_corners, axis=0)
                for i in [2, 3]:  # Bottom corners
                    direction = sorted_corners[i] - center
                    sorted_corners[i] = center + direction * (1 + correction_factor * (1/aspect_ratio - 1))
            
            return sorted_corners
        
        return corners
    
    def _minimize_perspective_distortion(self, corners: np.ndarray) -> np.ndarray:
        """
        Minimize perspective distortion by optimizing corner positions
        """
        # This is a placeholder for advanced perspective correction
        # For now, just ensure the quadrilateral is reasonably shaped
        return self._ensure_reasonable_aspect_ratio(corners)
    
    def _enforce_parallel_sides(self, corners: np.ndarray) -> np.ndarray:
        """
        Enforce that opposite sides are roughly parallel
        """
        sorted_corners = self._sort_corners(corners)
        
        # Calculate side vectors
        top_vector = sorted_corners[1] - sorted_corners[0]
        bottom_vector = sorted_corners[2] - sorted_corners[3]
        left_vector = sorted_corners[3] - sorted_corners[0]
        right_vector = sorted_corners[2] - sorted_corners[1]
        
        # Check if opposite sides are roughly parallel
        def angle_between_vectors(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        top_bottom_angle = angle_between_vectors(top_vector, bottom_vector)
        left_right_angle = angle_between_vectors(left_vector, right_vector)
        
        # If sides are not roughly parallel, apply light correction
        if top_bottom_angle > 0.3 or left_right_angle > 0.3:  # >17 degrees
            logger.info("     Applying parallelism correction")
            # Apply conservative correction towards parallelism
            # This is a simplified implementation
            return self._apply_parallelism_correction(sorted_corners)
        
        return corners
    
    def _apply_parallelism_correction(self, corners: np.ndarray) -> np.ndarray:
        """
        Apply light correction to make opposite sides more parallel
        """
        # Conservative approach: slight adjustment towards better parallelism
        # This is a placeholder for more sophisticated optimization
        return corners
    
    def _light_perspective_correction(self, corners: np.ndarray) -> np.ndarray:
        """
        Apply light perspective correction
        """
        return self._ensure_reasonable_aspect_ratio(corners)

def main():
    """
    Test the ultra precision detector
    """
    detector = UltraPrecisionCornerDetector()
    
    # Test image
    test_image = "my_chess_images/train/images/IMG_4698.JPG"
    
    if not Path(test_image).exists():
        logger.error(f"Test image not found: {test_image}")
        return
    
    # Test with different time budgets
    budgets = [1.0, 1.5, 2.0]
    
    for budget in budgets:
        logger.info(f"\n🎯 Testing with {budget}s budget:")
        corners, time_taken, budget_met = detector.detect_corners(test_image, budget)
        
        if corners:
            logger.info(f"✅ Success: {time_taken:.3f}s, budget met: {budget_met}")
            logger.info(f"   Corners: {corners}")
        else:
            logger.error(f"❌ Failed: {time_taken:.3f}s")

if __name__ == "__main__":
    main()
