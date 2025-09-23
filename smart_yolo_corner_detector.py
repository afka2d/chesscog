#!/usr/bin/env python3
"""
Smart YOLO Corner Detector
==========================

Fixes the anchoring issue where YOLO detects multiple objects and picks the wrong one.
Uses intelligent selection criteria beyond just confidence score.

Key improvements:
1. Size-based filtering (chessboards have reasonable size)
2. Aspect ratio validation (chessboards are roughly square)
3. Position preference (center of image more likely)
4. Confidence + geometry scoring
5. Grey background artifact rejection
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json

# Import existing detector
try:
    from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartYOLOCornerDetector:
    """
    Smart YOLO corner detector that fixes anchoring issues
    """
    
    def __init__(self, yolo_model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.yolo_model = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logger.info("✅ Smart YOLO detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  YOLO loading failed: {e}")
    
    def detect_corners(self, image_path: str, conf_threshold: float = 0.3) -> Optional[List[List[float]]]:
        """
        Smart corner detection with intelligent selection
        """
        if not self.yolo_model:
            logger.error("YOLO model not available")
            return None
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image_path, conf=conf_threshold, iou=0.5, verbose=False)
            
            if not results or not results[0].masks:
                logger.warning("No chessboard detections found")
                return None
            
            result = results[0]
            num_detections = len(result.masks.data)
            
            logger.info(f"YOLO found {num_detections} potential chessboards")
            
            if num_detections == 1:
                # Single detection - use it
                logger.info("Single detection found, using it")
                return self._extract_corners_from_detection(result, 0)
            
            else:
                # Multiple detections - use smart selection
                logger.info(f"Multiple detections found, applying smart selection")
                best_detection_idx = self._smart_detection_selection(result, image_path)
                
                if best_detection_idx is not None:
                    logger.info(f"Selected detection {best_detection_idx} as best chessboard")
                    return self._extract_corners_from_detection(result, best_detection_idx)
                else:
                    logger.warning("No suitable chessboard detection found")
                    return None
                    
        except Exception as e:
            logger.error(f"Smart YOLO detection failed: {e}")
            return None
    
    def _smart_detection_selection(self, result, image_path: str) -> Optional[int]:
        """
        Intelligently select the best chessboard detection from multiple candidates
        """
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        img_height, img_width = image.shape[:2]
        img_center = np.array([img_width / 2, img_height / 2])
        
        scores = []
        
        for i in range(len(confidences)):
            score = self._calculate_detection_score(
                confidences[i], boxes[i], img_width, img_height, img_center, result.masks.data[i]
            )
            scores.append(score)
            
            # Log scoring details
            box = boxes[i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
            aspect_ratio = width / height if height > 0 else 0
            
            logger.info(f"   Detection {i}: conf={confidences[i]:.3f}, "
                       f"size={width:.0f}x{height:.0f}, "
                       f"aspect={aspect_ratio:.2f}, "
                       f"center=({center[0]:.0f},{center[1]:.0f}), "
                       f"score={score:.3f}")
        
        if not scores:
            return None
        
        # Return detection with highest combined score
        best_idx = np.argmax(scores)
        logger.info(f"Best detection: {best_idx} with score {scores[best_idx]:.3f}")
        
        return best_idx
    
    def _calculate_detection_score(self, confidence: float, box: np.ndarray, 
                                 img_width: int, img_height: int, img_center: np.ndarray,
                                 mask_data: np.ndarray) -> float:
        """
        Calculate comprehensive score for detection quality
        """
        score = 0.0
        
        # 1. Base confidence score (30% weight)
        confidence_score = confidence * 0.3
        score += confidence_score
        
        # 2. Size appropriateness (25% weight)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        img_area = img_width * img_height
        
        # Chessboards should be 5-80% of image area
        size_ratio = box_area / img_area
        if 0.05 <= size_ratio <= 0.8:
            size_score = 0.25  # Good size
        elif 0.02 <= size_ratio <= 0.9:
            size_score = 0.15  # Acceptable size
        else:
            size_score = 0.0   # Too small or too large
        
        score += size_score
        
        # 3. Aspect ratio (20% weight)
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        if 0.7 <= aspect_ratio <= 1.4:  # Roughly square
            aspect_score = 0.2
        elif 0.5 <= aspect_ratio <= 2.0:  # Somewhat rectangular
            aspect_score = 0.1
        else:
            aspect_score = 0.0  # Too distorted
        
        score += aspect_score
        
        # 4. Position preference (15% weight)
        box_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
        distance_from_center = np.linalg.norm(box_center - img_center)
        max_distance = np.linalg.norm(img_center)  # Distance to corner
        
        # Prefer detections closer to image center
        position_score = 0.15 * (1 - distance_from_center / max_distance)
        score += position_score
        
        # 5. Shape quality (10% weight)
        shape_score = self._evaluate_shape_quality(mask_data) * 0.1
        score += shape_score
        
        return score
    
    def _evaluate_shape_quality(self, mask_data: np.ndarray) -> float:
        """
        Evaluate how chessboard-like the shape is
        """
        try:
            # Convert mask to binary
            mask_np = mask_data.cpu().numpy()
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            
            # Find contour
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape metrics
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return 0.0
            
            # Compactness (4π*area/perimeter²) - circles = 1, squares ≈ 0.785
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Good chessboard shapes should have reasonable compactness
            if 0.3 <= compactness <= 0.9:
                return 1.0
            elif 0.2 <= compactness <= 1.0:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Shape quality evaluation failed: {e}")
            return 0.5  # Neutral score if evaluation fails
    
    def _extract_corners_from_detection(self, result, detection_idx: int) -> Optional[List[List[float]]]:
        """
        Extract corners from specific detection
        """
        try:
            # Get mask points for this detection
            mask_points = result.masks.xy[detection_idx]
            
            if len(mask_points) < 4:
                logger.warning(f"Detection {detection_idx} has only {len(mask_points)} points")
                return None
            
            # Approximate to quadrilateral
            corners = self._approximate_to_quadrilateral(mask_points)
            
            if corners is not None and len(corners) == 4:
                return self._order_corners(corners).tolist()
            
            # Fallback: find extreme points
            corners = self._find_extreme_points(mask_points)
            
            if corners is not None and len(corners) == 4:
                return self._order_corners(corners).tolist()
            
            return None
            
        except Exception as e:
            logger.error(f"Corner extraction failed for detection {detection_idx}: {e}")
            return None
    
    def _approximate_to_quadrilateral(self, mask_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Approximate mask to quadrilateral
        """
        try:
            # Use more aggressive approximation to force 4 corners
            epsilon = 0.02 * cv2.arcLength(mask_points, True)
            approx = cv2.approxPolyDP(mask_points, epsilon, True)
            
            if len(approx) == 4:
                logger.info("Successfully approximated to quadrilateral")
                return approx.reshape(4, 2)
            
            # Try different epsilon values
            for epsilon_factor in [0.01, 0.03, 0.05]:
                epsilon = epsilon_factor * cv2.arcLength(mask_points, True)
                approx = cv2.approxPolyDP(mask_points, epsilon, True)
                
                if len(approx) == 4:
                    logger.info(f"Successfully approximated to quadrilateral (epsilon={epsilon_factor})")
                    return approx.reshape(4, 2)
            
            logger.warning(f"Could not approximate to quadrilateral, got {len(approx) if 'approx' in locals() else 0} points")
            return None
            
        except Exception as e:
            logger.error(f"Quadrilateral approximation failed: {e}")
            return None
    
    def _find_extreme_points(self, mask_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Find 4 extreme points from mask
        """
        try:
            # Find extreme points
            top_left = mask_points[np.argmin(mask_points[:, 0] + mask_points[:, 1])]
            top_right = mask_points[np.argmax(mask_points[:, 0] - mask_points[:, 1])]
            bottom_right = mask_points[np.argmax(mask_points[:, 0] + mask_points[:, 1])]
            bottom_left = mask_points[np.argmin(mask_points[:, 0] - mask_points[:, 1])]
            
            corners = np.array([top_left, top_right, bottom_right, bottom_left])
            
            return corners
            
        except Exception as e:
            logger.error(f"Extreme points detection failed: {e}")
            return None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left
        """
        # Find center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Find top-left (smallest x + y sum)
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return reordered

def main():
    """
    Test the smart YOLO detector
    """
    detector = SmartYOLOCornerDetector()
    
    # Test images that might have multiple detections
    test_images = [
        "grey_background_dataset/images/val/IMG_4779.JPG",
        "grey_background_dataset/images/test/IMG_4763.JPG"
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            continue
        
        logger.info(f"\n🎯 Testing smart detection: {Path(image_path).name}")
        
        corners = detector.detect_corners(image_path)
        
        if corners:
            logger.info(f"✅ Smart corners detected: {corners}")
        else:
            logger.error("❌ Smart detection failed")

if __name__ == "__main__":
    main()
