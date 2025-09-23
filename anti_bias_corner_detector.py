#!/usr/bin/env python3
"""
Anti-Bias Corner Detector
=========================

Specifically designed to fix the grey background training bias issue.
Addresses the problem where YOLO anchors to wrong objects due to training on grey backgrounds.

Key strategies:
1. Grey background artifact detection and rejection
2. Chessboard vs non-chessboard discrimination  
3. Size and geometry validation
4. Multiple detection ranking with domain knowledge
5. Confidence recalibration based on context
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

# Import existing detector
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiBiasCornerDetector:
    """
    Corner detector specifically designed to handle grey background training bias
    """
    
    def __init__(self, yolo_model_path="yolo_training_runs/yolo_chessboard_v1/weights/best.pt"):
        self.yolo_model = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                logger.info("✅ Anti-bias YOLO detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  YOLO loading failed: {e}")
    
    def detect_corners(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Detect corners with anti-bias logic
        """
        if not self.yolo_model:
            logger.error("YOLO model not available")
            return None
        
        try:
            # Run YOLO detection with lower confidence to catch more candidates
            results = self.yolo_model(image_path, conf=0.2, iou=0.5, verbose=False)
            
            if not results or not results[0].masks:
                logger.warning("No detections found")
                return None
            
            result = results[0]
            num_detections = len(result.masks.data)
            
            logger.info(f"🔍 Found {num_detections} potential detections")
            
            # Analyze each detection
            detection_analysis = self._analyze_all_detections(result, image_path)
            
            # Filter out grey background artifacts
            filtered_detections = self._filter_grey_background_artifacts(detection_analysis, image_path)
            
            if not filtered_detections:
                logger.warning("No valid chessboard detections after filtering")
                return None
            
            # Select best chessboard detection
            best_detection = self._select_best_chessboard(filtered_detections)
            
            if best_detection:
                logger.info(f"✅ Selected detection {best_detection['index']} as best chessboard")
                return self._extract_corners_from_detection(result, best_detection['index'])
            else:
                logger.warning("No suitable chessboard found")
                return None
                
        except Exception as e:
            logger.error(f"Anti-bias detection failed: {e}")
            return None
    
    def _analyze_all_detections(self, result, image_path: str) -> List[Dict]:
        """
        Analyze all detections to understand what YOLO is seeing
        """
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # Load image for analysis
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        
        detections = []
        
        for i in range(len(confidences)):
            box = boxes[i]
            mask = result.masks.data[i].cpu().numpy()
            
            analysis = {
                'index': i,
                'confidence': confidences[i],
                'box': box,
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'area_ratio': ((box[2] - box[0]) * (box[3] - box[1])) / (img_width * img_height),
                'aspect_ratio': (box[2] - box[0]) / (box[3] - box[1]) if (box[3] - box[1]) > 0 else 0,
                'center': [(box[0] + box[2])/2, (box[1] + box[3])/2],
                'is_grey_artifact': self._is_grey_background_artifact(image, box),
                'is_reasonable_chessboard': self._is_reasonable_chessboard(box, img_width, img_height),
                'shape_quality': self._evaluate_shape_quality(mask)
            }
            
            detections.append(analysis)
            
            logger.info(f"   Detection {i}: conf={analysis['confidence']:.3f}, "
                       f"area_ratio={analysis['area_ratio']:.3f}, "
                       f"aspect={analysis['aspect_ratio']:.2f}, "
                       f"grey_artifact={analysis['is_grey_artifact']}, "
                       f"reasonable_board={analysis['is_reasonable_chessboard']}")
        
        return detections
    
    def _is_grey_background_artifact(self, image: np.ndarray, box: np.ndarray) -> bool:
        """
        Detect if this detection is likely a grey background artifact
        """
        try:
            # Extract the region
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return True  # Invalid region
            
            # Convert to HSV for better color analysis
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate color characteristics
            mean_saturation = np.mean(region_hsv[:, :, 1])
            mean_value = np.mean(region_hsv[:, :, 2])
            
            # Grey background artifacts typically have:
            # - Low saturation (grey/monochrome)
            # - Uniform color distribution
            # - Specific brightness range
            
            is_low_saturation = mean_saturation < 30  # Very grey
            is_medium_brightness = 80 < mean_value < 200  # Not too dark/bright
            
            # Calculate color uniformity
            color_std = np.std(region_hsv[:, :, 2])
            is_uniform = color_std < 20  # Very uniform color
            
            # Check if it's likely a grey background artifact
            is_artifact = is_low_saturation and is_medium_brightness and is_uniform
            
            if is_artifact:
                logger.info(f"     Detected grey artifact: sat={mean_saturation:.1f}, "
                           f"brightness={mean_value:.1f}, uniformity={color_std:.1f}")
            
            return is_artifact
            
        except Exception as e:
            logger.warning(f"Grey artifact detection failed: {e}")
            return False
    
    def _is_reasonable_chessboard(self, box: np.ndarray, img_width: int, img_height: int) -> bool:
        """
        Check if detection has reasonable chessboard characteristics
        """
        width = box[2] - box[0]
        height = box[3] - box[1]
        area_ratio = (width * height) / (img_width * img_height)
        aspect_ratio = width / height if height > 0 else 0
        
        # Reasonable chessboard criteria
        size_ok = 0.1 <= area_ratio <= 0.9  # 10-90% of image
        aspect_ok = 0.5 <= aspect_ratio <= 2.0  # Not too distorted
        not_tiny = width > 100 and height > 100  # Minimum size
        not_huge = width < img_width * 0.95 and height < img_height * 0.95  # Not entire image
        
        return size_ok and aspect_ok and not_tiny and not_huge
    
    def _evaluate_shape_quality(self, mask_data: np.ndarray) -> float:
        """
        Evaluate shape quality for chessboard likelihood
        """
        try:
            mask_np = mask_data.cpu().numpy()
            
            # Calculate basic shape metrics
            non_zero_pixels = np.count_nonzero(mask_np)
            total_pixels = mask_np.size
            
            if total_pixels == 0:
                return 0.0
            
            fill_ratio = non_zero_pixels / total_pixels
            
            # Good chessboard masks should have reasonable fill ratio
            if 0.1 <= fill_ratio <= 0.8:
                return 1.0
            elif 0.05 <= fill_ratio <= 0.9:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Shape quality evaluation failed: {e}")
            return 0.5
    
    def _filter_grey_background_artifacts(self, detections: List[Dict], image_path: str) -> List[Dict]:
        """
        Filter out detections that are likely grey background artifacts
        """
        filtered = []
        
        for detection in detections:
            # Skip if clearly a grey artifact
            if detection['is_grey_artifact']:
                logger.info(f"   Filtering out detection {detection['index']}: grey background artifact")
                continue
            
            # Skip if not reasonable chessboard
            if not detection['is_reasonable_chessboard']:
                logger.info(f"   Filtering out detection {detection['index']}: unreasonable chessboard")
                continue
            
            # Skip if confidence is too low (likely noise)
            if detection['confidence'] < 0.3:
                logger.info(f"   Filtering out detection {detection['index']}: low confidence {detection['confidence']:.3f}")
                continue
            
            filtered.append(detection)
        
        logger.info(f"✅ Filtered {len(detections)} → {len(filtered)} valid detections")
        return filtered
    
    def _select_best_chessboard(self, filtered_detections: List[Dict]) -> Optional[Dict]:
        """
        Select the best chessboard from filtered detections
        """
        if not filtered_detections:
            return None
        
        if len(filtered_detections) == 1:
            return filtered_detections[0]
        
        # Calculate composite scores
        for detection in filtered_detections:
            score = 0.0
            
            # Confidence (40% weight)
            score += detection['confidence'] * 0.4
            
            # Size appropriateness (30% weight)
            if 0.2 <= detection['area_ratio'] <= 0.7:  # Good size
                score += 0.3
            elif 0.1 <= detection['area_ratio'] <= 0.8:  # Acceptable
                score += 0.15
            
            # Aspect ratio (20% weight)
            if 0.8 <= detection['aspect_ratio'] <= 1.25:  # Nearly square
                score += 0.2
            elif 0.6 <= detection['aspect_ratio'] <= 1.5:  # Somewhat square
                score += 0.1
            
            # Shape quality (10% weight)
            score += detection['shape_quality'] * 0.1
            
            detection['composite_score'] = score
            
            logger.info(f"   Detection {detection['index']}: composite score = {score:.3f}")
        
        # Return detection with highest composite score
        best_detection = max(filtered_detections, key=lambda d: d['composite_score'])
        
        return best_detection
    
    def _extract_corners_from_detection(self, result, detection_idx: int) -> Optional[List[List[float]]]:
        """
        Extract corners from the selected detection
        """
        try:
            mask_points = result.masks.xy[detection_idx]
            
            # Approximate to quadrilateral
            corners = self._approximate_to_quadrilateral(mask_points)
            
            if corners is not None and len(corners) == 4:
                ordered_corners = self._order_corners(corners)
                return ordered_corners.tolist()
            
            # Fallback to extreme points
            corners = self._find_extreme_points(mask_points)
            
            if corners is not None and len(corners) == 4:
                ordered_corners = self._order_corners(corners)
                return ordered_corners.tolist()
            
            return None
            
        except Exception as e:
            logger.error(f"Corner extraction failed: {e}")
            return None
    
    def _approximate_to_quadrilateral(self, mask_points: np.ndarray) -> Optional[np.ndarray]:
        """Approximate mask to quadrilateral"""
        try:
            epsilon = 0.02 * cv2.arcLength(mask_points, True)
            approx = cv2.approxPolyDP(mask_points, epsilon, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2)
            
            # Try different epsilon values
            for epsilon_factor in [0.01, 0.03, 0.05]:
                epsilon = epsilon_factor * cv2.arcLength(mask_points, True)
                approx = cv2.approxPolyDP(mask_points, epsilon, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2)
            
            return None
            
        except Exception as e:
            logger.error(f"Quadrilateral approximation failed: {e}")
            return None
    
    def _find_extreme_points(self, mask_points: np.ndarray) -> Optional[np.ndarray]:
        """Find 4 extreme points"""
        try:
            top_left = mask_points[np.argmin(mask_points[:, 0] + mask_points[:, 1])]
            top_right = mask_points[np.argmax(mask_points[:, 0] - mask_points[:, 1])]
            bottom_right = mask_points[np.argmax(mask_points[:, 0] + mask_points[:, 1])]
            bottom_left = mask_points[np.argmin(mask_points[:, 0] - mask_points[:, 1])]
            
            return np.array([top_left, top_right, bottom_right, bottom_left])
            
        except Exception as e:
            logger.error(f"Extreme points failed: {e}")
            return None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners consistently"""
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)
        
        return reordered

def main():
    """Test the anti-bias detector"""
    detector = AntiBiasCornerDetector()
    
    test_images = [
        "grey_background_dataset/images/val/IMG_4779.JPG",
        "grey_background_dataset/images/test/IMG_4763.JPG"
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            continue
        
        logger.info(f"\n🎯 Testing anti-bias detection: {Path(image_path).name}")
        corners = detector.detect_corners(image_path)
        
        if corners:
            logger.info(f"✅ Anti-bias corners: {corners}")
        else:
            logger.error("❌ Anti-bias detection failed")

if __name__ == "__main__":
    main()
