#!/usr/bin/env python3
"""
Robust Chessboard Detector
==========================

Comprehensive solution to fix the anchoring/grey background bias issue.

PROBLEM ANALYSIS:
- YOLO finds 6-12 detections instead of 1
- Picks highest confidence, which may be grey background artifact
- Training bias: all training images have grey backgrounds
- YOLO learned to detect grey rectangles as chessboards

SOLUTION STRATEGY:
1. Confidence threshold adjustment
2. Multi-criteria detection ranking
3. Chessboard-specific validation
4. Grey background artifact rejection
5. Size and position filtering
6. Geometric plausibility checks
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustChessboardDetector:
    """
    Robust chessboard detector that handles multiple false detections
    """
    
    def __init__(self, yolo_model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.yolo_model = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logger.info("✅ Robust chessboard detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  YOLO loading failed: {e}")
    
    def detect_corners(self, image_path: str, debug: bool = True) -> Optional[List[List[float]]]:
        """
        Robust corner detection with comprehensive filtering
        """
        if not self.yolo_model:
            logger.error("YOLO model not available")
            return None
        
        logger.info(f"🔍 Robust chessboard detection: {Path(image_path).name}")
        
        try:
            # Run YOLO with lower confidence to see all detections
            results = self.yolo_model(image_path, conf=0.1, iou=0.3, verbose=False)
            
            if not results or not results[0].masks:
                logger.warning("No detections found")
                return None
            
            result = results[0]
            num_detections = len(result.masks.data)
            
            logger.info(f"🔍 Raw YOLO found {num_detections} detections")
            
            if num_detections == 1:
                # Single detection - validate it's reasonable
                if self._validate_single_detection(result, 0, image_path):
                    logger.info("✅ Single detection validated")
                    return self._extract_corners_from_detection(result, 0)
                else:
                    logger.warning("❌ Single detection failed validation")
                    return None
            
            else:
                # Multiple detections - apply comprehensive filtering
                logger.info(f"🔍 Multiple detections found, applying smart filtering...")
                best_detection_idx = self._comprehensive_detection_selection(result, image_path, debug)
                
                if best_detection_idx is not None:
                    logger.info(f"✅ Selected detection {best_detection_idx} as best chessboard")
                    return self._extract_corners_from_detection(result, best_detection_idx)
                else:
                    logger.warning("❌ No suitable chessboard found after filtering")
                    return None
                    
        except Exception as e:
            logger.error(f"Robust detection failed: {e}")
            return None
    
    def _validate_single_detection(self, result, detection_idx: int, image_path: str) -> bool:
        """
        Validate that a single detection is actually a chessboard
        """
        try:
            box = result.boxes.xyxy[detection_idx].cpu().numpy()
            confidence = result.boxes.conf[detection_idx].cpu().numpy()
            
            # Load image for validation
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]
            
            # Basic validation criteria
            width = box[2] - box[0]
            height = box[3] - box[1]
            area_ratio = (width * height) / (img_width * img_height)
            aspect_ratio = width / height if height > 0 else 0
            
            # Validation checks
            confidence_ok = confidence > 0.5  # Reasonable confidence
            size_ok = 0.1 <= area_ratio <= 0.8  # Reasonable size
            aspect_ok = 0.6 <= aspect_ratio <= 1.7  # Roughly square-ish
            not_edge = self._not_on_image_edge(box, img_width, img_height)
            
            is_valid = confidence_ok and size_ok and aspect_ok and not_edge
            
            logger.info(f"   Single detection validation: conf={confidence:.3f}, "
                       f"size={area_ratio:.3f}, aspect={aspect_ratio:.2f}, "
                       f"not_edge={not_edge}, valid={is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Single detection validation failed: {e}")
            return False
    
    def _comprehensive_detection_selection(self, result, image_path: str, debug: bool = True) -> Optional[int]:
        """
        Comprehensive selection from multiple detections
        """
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # Load image
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        
        candidates = []
        
        for i in range(len(confidences)):
            box = boxes[i]
            confidence = confidences[i]
            
            # Calculate detection characteristics
            width = box[2] - box[0]
            height = box[3] - box[1]
            area_ratio = (width * height) / (img_width * img_height)
            aspect_ratio = width / height if height > 0 else 0
            
            # Comprehensive scoring
            score = self._calculate_comprehensive_score(
                confidence, box, area_ratio, aspect_ratio, image, img_width, img_height
            )
            
            candidate = {
                'index': i,
                'confidence': confidence,
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio,
                'score': score,
                'box': box
            }
            
            candidates.append(candidate)
            
            if debug:
                logger.info(f"   Detection {i}: conf={confidence:.3f}, "
                           f"area={area_ratio:.3f}, aspect={aspect_ratio:.2f}, "
                           f"score={score:.3f}")
        
        # Filter candidates
        valid_candidates = [c for c in candidates if c['score'] > 0.3]  # Minimum threshold
        
        if not valid_candidates:
            logger.warning("No candidates passed minimum score threshold")
            return None
        
        # Sort by score and return best
        valid_candidates.sort(key=lambda c: c['score'], reverse=True)
        best_candidate = valid_candidates[0]
        
        logger.info(f"🏆 Best candidate: detection {best_candidate['index']} "
                   f"(score: {best_candidate['score']:.3f})")
        
        return best_candidate['index']
    
    def _calculate_comprehensive_score(self, confidence: float, box: np.ndarray, 
                                     area_ratio: float, aspect_ratio: float,
                                     image: np.ndarray, img_width: int, img_height: int) -> float:
        """
        Calculate comprehensive score for chessboard likelihood
        """
        score = 0.0
        
        # 1. Confidence component (25% weight)
        confidence_score = min(confidence, 1.0) * 0.25
        score += confidence_score
        
        # 2. Size appropriateness (25% weight)
        if 0.15 <= area_ratio <= 0.7:  # Ideal chessboard size
            size_score = 0.25
        elif 0.1 <= area_ratio <= 0.8:  # Acceptable size
            size_score = 0.15
        elif 0.05 <= area_ratio <= 0.9:  # Marginal size
            size_score = 0.1
        else:
            size_score = 0.0  # Too small or too large
        
        score += size_score
        
        # 3. Aspect ratio (20% weight)
        if 0.8 <= aspect_ratio <= 1.25:  # Nearly square (ideal)
            aspect_score = 0.2
        elif 0.6 <= aspect_ratio <= 1.5:  # Somewhat square
            aspect_score = 0.15
        elif 0.5 <= aspect_ratio <= 2.0:  # Rectangular but reasonable
            aspect_score = 0.1
        else:
            aspect_score = 0.0  # Too distorted
        
        score += aspect_score
        
        # 4. Position preference (15% weight)
        box_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
        img_center = np.array([img_width/2, img_height/2])
        distance_from_center = np.linalg.norm(box_center - img_center)
        max_distance = np.linalg.norm(img_center)
        
        # Prefer detections closer to center
        position_score = 0.15 * (1 - distance_from_center / max_distance)
        score += position_score
        
        # 5. Grey background artifact penalty (15% weight)
        is_grey_artifact = self._is_grey_background_artifact(image, box)
        if is_grey_artifact:
            grey_penalty = -0.3  # Heavy penalty for grey artifacts
        else:
            grey_penalty = 0.15  # Bonus for non-grey regions
        
        score += grey_penalty
        
        return max(score, 0.0)  # Ensure non-negative
    
    def _is_grey_background_artifact(self, image: np.ndarray, box: np.ndarray) -> bool:
        """
        Detect grey background artifacts
        """
        try:
            # Extract region
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return True  # Invalid region
            
            region = image[y1:y2, x1:x2]
            
            # Convert to HSV for better color analysis
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate color statistics
            mean_saturation = np.mean(region_hsv[:, :, 1])
            std_saturation = np.std(region_hsv[:, :, 1])
            mean_value = np.mean(region_hsv[:, :, 2])
            std_value = np.std(region_hsv[:, :, 2])
            
            # Grey background characteristics:
            # - Very low saturation (grey/monochrome)
            # - Low color variation
            # - Medium brightness
            
            is_grey = mean_saturation < 25  # Very desaturated
            is_uniform = std_saturation < 15 and std_value < 30  # Very uniform
            is_medium_bright = 60 < mean_value < 180  # Not too dark/bright
            
            is_artifact = is_grey and is_uniform and is_medium_bright
            
            if is_artifact:
                logger.info(f"     🚫 Grey artifact detected: sat={mean_saturation:.1f}, "
                           f"uniformity={std_value:.1f}")
            
            return is_artifact
            
        except Exception as e:
            logger.warning(f"Grey artifact detection failed: {e}")
            return False
    
    def _not_on_image_edge(self, box: np.ndarray, img_width: int, img_height: int) -> bool:
        """
        Check that detection is not on the edge of the image
        """
        edge_margin = 20  # pixels
        
        x1, y1, x2, y2 = box
        
        # Check if any edge of the box is too close to image edge
        too_close_to_left = x1 < edge_margin
        too_close_to_top = y1 < edge_margin
        too_close_to_right = x2 > (img_width - edge_margin)
        too_close_to_bottom = y2 > (img_height - edge_margin)
        
        is_on_edge = too_close_to_left or too_close_to_top or too_close_to_right or too_close_to_bottom
        
        return not is_on_edge
    
    def _extract_corners_from_detection(self, result, detection_idx: int) -> Optional[List[List[float]]]:
        """
        Extract corners from selected detection
        """
        try:
            mask_points = result.masks.xy[detection_idx]
            
            # Approximate to quadrilateral
            corners = self._approximate_to_quadrilateral(mask_points)
            
            if corners is not None and len(corners) == 4:
                ordered_corners = self._order_corners(corners)
                return ordered_corners.tolist()
            
            return None
            
        except Exception as e:
            logger.error(f"Corner extraction failed: {e}")
            return None
    
    def _approximate_to_quadrilateral(self, mask_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Approximate mask to quadrilateral with multiple strategies
        """
        try:
            # Strategy 1: Standard approximation
            epsilon = 0.02 * cv2.arcLength(mask_points, True)
            approx = cv2.approxPolyDP(mask_points, epsilon, True)
            
            if len(approx) == 4:
                logger.info("✅ Standard quadrilateral approximation successful")
                return approx.reshape(4, 2)
            
            # Strategy 2: More aggressive approximation
            for epsilon_factor in [0.03, 0.05, 0.08]:
                epsilon = epsilon_factor * cv2.arcLength(mask_points, True)
                approx = cv2.approxPolyDP(mask_points, epsilon, True)
                
                if len(approx) == 4:
                    logger.info(f"✅ Aggressive approximation successful (ε={epsilon_factor})")
                    return approx.reshape(4, 2)
            
            # Strategy 3: Convex hull + extreme points
            logger.info("Standard approximation failed, using convex hull method")
            hull = cv2.convexHull(mask_points)
            corners = self._find_four_corners_from_hull(hull)
            
            if corners is not None:
                logger.info("✅ Convex hull method successful")
                return corners
            
            logger.warning("All quadrilateral approximation methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Quadrilateral approximation failed: {e}")
            return None
    
    def _find_four_corners_from_hull(self, hull: np.ndarray) -> Optional[np.ndarray]:
        """
        Find 4 corners from convex hull
        """
        try:
            hull = hull.reshape(-1, 2)
            
            # Find extreme points
            top_left = hull[np.argmin(hull[:, 0] + hull[:, 1])]
            top_right = hull[np.argmax(hull[:, 0] - hull[:, 1])]
            bottom_right = hull[np.argmax(hull[:, 0] + hull[:, 1])]
            bottom_left = hull[np.argmin(hull[:, 0] - hull[:, 1])]
            
            corners = np.array([top_left, top_right, bottom_right, bottom_left])
            
            # Validate corners are reasonable
            if self._validate_corner_geometry(corners):
                return corners
            else:
                logger.warning("Corner geometry validation failed")
                return None
                
        except Exception as e:
            logger.error(f"Hull corner extraction failed: {e}")
            return None
    
    def _validate_corner_geometry(self, corners: np.ndarray) -> bool:
        """
        Validate that corners form a reasonable chessboard shape
        """
        try:
            # Check that corners form a convex quadrilateral
            area = cv2.contourArea(corners.astype(np.float32))
            
            if area < 5000:  # Too small
                return False
            
            # Check aspect ratio
            sorted_corners = self._order_corners(corners)
            width1 = np.linalg.norm(sorted_corners[1] - sorted_corners[0])
            width2 = np.linalg.norm(sorted_corners[2] - sorted_corners[3])
            height1 = np.linalg.norm(sorted_corners[3] - sorted_corners[0])
            height2 = np.linalg.norm(sorted_corners[2] - sorted_corners[1])
            
            avg_width = (width1 + width2) / 2
            avg_height = (height1 + height2) / 2
            
            if avg_height == 0:
                return False
            
            aspect_ratio = avg_width / avg_height
            
            # Chessboards should be roughly square
            if 0.5 <= aspect_ratio <= 2.0:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Geometry validation failed: {e}")
            return False
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently
        """
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return reordered

def main():
    """
    Test the robust chessboard detector
    """
    detector = RobustChessboardDetector()
    
    # Test with images that are known to have multiple detections
    test_images = [
        "grey_background_dataset/images/val/IMG_4779.JPG",
        "grey_background_dataset/images/test/IMG_4763.JPG",
        "grey_background_dataset/images/test/IMG_4785.JPG"
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            continue
        
        logger.info(f"\n" + "="*60)
        logger.info(f"🎯 TESTING: {Path(image_path).name}")
        logger.info("="*60)
        
        corners = detector.detect_corners(image_path, debug=True)
        
        if corners:
            logger.info(f"✅ ROBUST DETECTION SUCCESS")
            logger.info(f"   Corners: {corners}")
            
            # Load ground truth for comparison if available
            annotation_path = f"grey_background_dataset/annotations/train/{Path(image_path).name.replace('.JPG', '.json')}"
            if Path(annotation_path).exists():
                with open(annotation_path, 'r') as f:
                    gt_data = json.load(f)
                gt_corners = gt_data.get('corners')
                
                if gt_corners:
                    # Calculate error
                    gt_np = np.array(gt_corners)
                    pred_np = np.array(corners)
                    errors = np.linalg.norm(gt_np - pred_np, axis=1)
                    avg_error = np.mean(errors)
                    
                    logger.info(f"📊 Accuracy vs ground truth: {avg_error:.1f}px average error")
                    
                    if avg_error < 20:
                        logger.info("🏆 EXCELLENT accuracy!")
                    elif avg_error < 40:
                        logger.info("✅ GOOD accuracy")
                    else:
                        logger.info("⚠️  Needs improvement")
        else:
            logger.error("❌ ROBUST DETECTION FAILED")

if __name__ == "__main__":
    main()
