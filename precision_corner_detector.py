#!/usr/bin/env python3
"""
Precision Corner Detector
=========================

Ultra-precise corner detection using multiple refinement stages:
1. YOLO initial detection
2. OpenCV chessboard pattern matching
3. Harris corner detection in local regions
4. Sub-pixel refinement
5. Geometric validation and correction
6. Edge-based line fitting verification
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import time
import matplotlib.pyplot as plt

# Import YOLO detector
try:
    from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionCornerDetector:
    """
    Ultra-precise corner detection system
    """
    
    def __init__(self, enable_visualization: bool = False):
        self.yolo_detector = None
        self.enable_viz = enable_visualization
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = ImprovedYOLOCornerDetector()
                logger.info("✅ YOLO detector loaded")
            except Exception as e:
                logger.warning(f"⚠️  YOLO failed to load: {e}")
        
        # Detection parameters
        self.chessboard_size = (7, 7)  # Internal corners
        self.harris_params = {
            'blockSize': 2,
            'ksize': 3,
            'k': 0.04
        }
        
    def detect_corners_ultra_precise(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Ultra-precise corner detection pipeline
        """
        logger.info(f"🎯 Ultra-precise detection: {Path(image_path).name}")
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Pipeline stages
        stages_results = {}
        
        # Stage 1: YOLO initial detection
        yolo_corners = self._stage1_yolo_detection(image_path)
        stages_results['yolo'] = yolo_corners
        
        if yolo_corners is None:
            logger.warning("YOLO failed, trying full OpenCV pipeline")
            return self._full_opencv_pipeline(image)
        
        # Stage 2: Chessboard pattern refinement
        pattern_corners = self._stage2_chessboard_pattern(image, yolo_corners)
        stages_results['pattern'] = pattern_corners
        
        # Stage 3: Harris corner refinement
        harris_corners = self._stage3_harris_refinement(image, pattern_corners or yolo_corners)
        stages_results['harris'] = harris_corners
        
        # Stage 4: Sub-pixel refinement
        subpixel_corners = self._stage4_subpixel_refinement(image, harris_corners or pattern_corners or yolo_corners)
        stages_results['subpixel'] = subpixel_corners
        
        # Stage 5: Geometric validation and correction
        final_corners = self._stage5_geometric_validation(subpixel_corners or harris_corners or pattern_corners or yolo_corners, image.shape)
        stages_results['final'] = final_corners
        
        # Log the pipeline results
        self._log_pipeline_results(stages_results)
        
        if self.enable_viz:
            self._visualize_pipeline_stages(image, stages_results, image_path)
        
        return final_corners
    
    def _stage1_yolo_detection(self, image_path: str) -> Optional[List[List[float]]]:
        """Stage 1: YOLO detection"""
        if self.yolo_detector is None:
            return None
        
        try:
            corners = self.yolo_detector.detect_corners(image_path)
            if corners is not None:
                logger.info(f"   Stage 1 (YOLO): ✅ Success")
                return corners if isinstance(corners, list) else corners.tolist()
        except Exception as e:
            logger.warning(f"   Stage 1 (YOLO): ❌ Failed - {e}")
        
        return None
    
    def _stage2_chessboard_pattern(self, image: np.ndarray, initial_corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 2: OpenCV chessboard pattern matching"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create focused ROI around YOLO detection
            corners_np = np.array(initial_corners, dtype=np.float32)
            
            # Expand ROI for pattern detection
            margin = 150
            x_min = max(0, int(np.min(corners_np[:, 0]) - margin))
            y_min = max(0, int(np.min(corners_np[:, 1]) - margin))
            x_max = min(gray.shape[1], int(np.max(corners_np[:, 0]) + margin))
            y_max = min(gray.shape[0], int(np.max(corners_np[:, 1]) + margin))
            
            roi = gray[y_min:y_max, x_min:x_max]
            
            # Enhanced preprocessing for better pattern detection
            roi = cv2.equalizeHist(roi)  # Improve contrast
            roi = cv2.GaussianBlur(roi, (3, 3), 0)  # Reduce noise
            
            # Try multiple chessboard detection strategies
            strategies = [
                # Strategy 1: Standard detection
                (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE, "standard"),
                # Strategy 2: Fast check
                (cv2.CALIB_CB_FAST_CHECK, "fast"),
                # Strategy 3: Adaptive + fast
                (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK, "adaptive_fast")
            ]
            
            for flags, strategy_name in strategies:
                found, corners_roi = cv2.findChessboardCorners(roi, self.chessboard_size, flags=flags)
                
                if found and corners_roi is not None:
                    logger.info(f"   Stage 2 (Pattern-{strategy_name}): ✅ Found {len(corners_roi)} corners")
                    
                    # Convert back to full image coordinates
                    corners_roi[:, :, 0] += x_min
                    corners_roi[:, :, 1] += y_min
                    
                    # Extract board corners
                    board_corners = self._extract_board_corners_from_pattern(corners_roi)
                    if board_corners is not None:
                        return board_corners.tolist()
            
            logger.info(f"   Stage 2 (Pattern): ❌ No pattern found")
            
        except Exception as e:
            logger.warning(f"   Stage 2 (Pattern): ❌ Failed - {e}")
        
        return None
    
    def _stage3_harris_refinement(self, image: np.ndarray, corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 3: Harris corner detection for local refinement"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            refined_corners = []
            
            for i, corner in enumerate(corners_np):
                # Create local region around each corner
                region_size = 50
                x, y = int(corner[0]), int(corner[1])
                
                x_start = max(0, x - region_size)
                y_start = max(0, y - region_size)
                x_end = min(gray.shape[1], x + region_size)
                y_end = min(gray.shape[0], y + region_size)
                
                region = gray[y_start:y_end, x_start:x_end]
                
                # Harris corner detection in local region
                harris_response = cv2.cornerHarris(region, **self.harris_params)
                
                # Find the strongest corner in the region
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris_response)
                
                if max_val > 0.01 * harris_response.max():  # Threshold for valid corner
                    # Convert back to full image coordinates
                    refined_x = x_start + max_loc[0]
                    refined_y = y_start + max_loc[1]
                    refined_corners.append([refined_x, refined_y])
                else:
                    # Keep original corner if no strong Harris response
                    refined_corners.append(corner.tolist())
            
            if len(refined_corners) == 4:
                logger.info(f"   Stage 3 (Harris): ✅ Refined all corners")
                return refined_corners
            
        except Exception as e:
            logger.warning(f"   Stage 3 (Harris): ❌ Failed - {e}")
        
        return None
    
    def _stage4_subpixel_refinement(self, image: np.ndarray, corners: List[List[float]]) -> Optional[List[List[float]]]:
        """Stage 4: Sub-pixel accuracy refinement"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners_np = np.array(corners, dtype=np.float32)
            
            # Sub-pixel refinement with tight criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            winSize = (15, 15)  # Larger window for better accuracy
            zeroZone = (-1, -1)
            
            refined_corners = cv2.cornerSubPix(gray, corners_np, winSize, zeroZone, criteria)
            
            # Validate refinement (corners shouldn't move too far)
            movements = np.linalg.norm(refined_corners - corners_np, axis=1)
            max_movement = np.max(movements)
            avg_movement = np.mean(movements)
            
            if max_movement < 30 and avg_movement < 15:  # Reasonable movement
                logger.info(f"   Stage 4 (Sub-pixel): ✅ Avg movement: {avg_movement:.1f}px")
                return refined_corners.tolist()
            else:
                logger.warning(f"   Stage 4 (Sub-pixel): ⚠️  Large movement: {max_movement:.1f}px")
                
        except Exception as e:
            logger.warning(f"   Stage 4 (Sub-pixel): ❌ Failed - {e}")
        
        return None
    
    def _stage5_geometric_validation(self, corners: List[List[float]], image_shape: Tuple[int, int, int]) -> Optional[List[List[float]]]:
        """Stage 5: Geometric validation and correction"""
        try:
            corners_np = np.array(corners, dtype=np.float32)
            
            # Reorder corners consistently
            ordered_corners = self._order_corners_clockwise(corners_np)
            
            # Validate geometry
            if not self._validate_chessboard_geometry(ordered_corners):
                logger.warning("   Stage 5 (Geometric): ⚠️  Invalid geometry detected")
                # Try to correct
                corrected_corners = self._correct_geometry(ordered_corners, image_shape)
                if corrected_corners is not None:
                    logger.info("   Stage 5 (Geometric): ✅ Geometry corrected")
                    return corrected_corners.tolist()
                else:
                    logger.warning("   Stage 5 (Geometric): ❌ Could not correct geometry")
                    return ordered_corners.tolist()  # Return best effort
            else:
                logger.info("   Stage 5 (Geometric): ✅ Valid geometry")
                return ordered_corners.tolist()
                
        except Exception as e:
            logger.warning(f"   Stage 5 (Geometric): ❌ Failed - {e}")
        
        return None
    
    def _extract_board_corners_from_pattern(self, internal_corners: np.ndarray) -> Optional[np.ndarray]:
        """Extract board corners from internal chessboard pattern"""
        try:
            # Reshape to 7x7 grid
            corners_grid = internal_corners.reshape(7, 7, 2)
            
            # Get corner internal points
            tl_internal = corners_grid[0, 0]
            tr_internal = corners_grid[0, 6]
            bl_internal = corners_grid[6, 0]
            br_internal = corners_grid[6, 6]
            
            # Calculate square dimensions
            square_width = (tr_internal[0] - tl_internal[0]) / 6
            square_height = (bl_internal[1] - tl_internal[1]) / 6
            
            # Extrapolate to board edges
            tl_board = tl_internal - [square_width, square_height]
            tr_board = tr_internal + [square_width, -square_height]
            br_board = br_internal + [square_width, square_height]
            bl_board = bl_internal - [square_width, -square_height]
            
            return np.array([tl_board, tr_board, br_board, bl_board])
            
        except Exception as e:
            logger.warning(f"Pattern corner extraction failed: {e}")
            return None
    
    def _order_corners_clockwise(self, corners: np.ndarray) -> np.ndarray:
        """Order corners clockwise starting from top-left"""
        # Find center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort by angle (clockwise from top-left)
        sorted_indices = np.argsort(angles)
        
        # Find top-left corner (smallest x+y sum)
        sums = corners[:, 0] + corners[:, 1]
        tl_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        start_idx = np.where(sorted_indices == tl_idx)[0][0]
        ordered_indices = np.roll(sorted_indices, -start_idx)
        
        return corners[ordered_indices]
    
    def _validate_chessboard_geometry(self, corners: np.ndarray) -> bool:
        """Validate that corners form a reasonable chessboard shape"""
        try:
            # Check aspect ratio (should be close to square)
            width = np.linalg.norm(corners[1] - corners[0])
            height = np.linalg.norm(corners[3] - corners[0])
            aspect_ratio = width / height
            
            if not (0.7 <= aspect_ratio <= 1.4):  # Reasonable aspect ratio
                return False
            
            # Check if quadrilateral is convex
            if not self._is_convex(corners):
                return False
            
            # Check if corners are roughly rectangular
            angles = self._calculate_internal_angles(corners)
            for angle in angles:
                if not (60 <= angle <= 120):  # Angles should be close to 90 degrees
                    return False
            
            return True
            
        except:
            return False
    
    def _is_convex(self, corners: np.ndarray) -> bool:
        """Check if quadrilateral is convex"""
        try:
            n = len(corners)
            cross_products = []
            
            for i in range(n):
                p1 = corners[i]
                p2 = corners[(i + 1) % n]
                p3 = corners[(i + 2) % n]
                
                v1 = p2 - p1
                v2 = p3 - p2
                cross = np.cross(v1, v2)
                cross_products.append(cross)
            
            # All cross products should have the same sign for convex polygon
            signs = [np.sign(cp) for cp in cross_products if abs(cp) > 1e-10]
            return len(set(signs)) <= 1
            
        except:
            return False
    
    def _calculate_internal_angles(self, corners: np.ndarray) -> List[float]:
        """Calculate internal angles of the quadrilateral"""
        angles = []
        n = len(corners)
        
        for i in range(n):
            p1 = corners[(i - 1) % n]
            p2 = corners[i]
            p3 = corners[(i + 1) % n]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            angles.append(angle)
        
        return angles
    
    def _correct_geometry(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Apply geometric correction to invalid corners"""
        try:
            h, w = image_shape[:2]
            
            # Project to ideal rectangle and back
            # This helps correct perspective distortion
            
            # Calculate current quadrilateral properties
            center = np.mean(corners, axis=0)
            
            # Calculate average side lengths
            side_lengths = []
            for i in range(4):
                side_length = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
                side_lengths.append(side_length)
            
            avg_side_length = np.mean(side_lengths)
            
            # Create ideal square centered at current center
            half_side = avg_side_length / 2
            ideal_corners = np.array([
                [center[0] - half_side, center[1] - half_side],  # TL
                [center[0] + half_side, center[1] - half_side],  # TR
                [center[0] + half_side, center[1] + half_side],  # BR
                [center[0] - half_side, center[1] + half_side]   # BL
            ])
            
            # Apply weighted average between current and ideal
            weight = 0.3  # 30% correction toward ideal
            corrected_corners = (1 - weight) * corners + weight * ideal_corners
            
            # Ensure corners are within image bounds
            corrected_corners[:, 0] = np.clip(corrected_corners[:, 0], 0, w - 1)
            corrected_corners[:, 1] = np.clip(corrected_corners[:, 1], 0, h - 1)
            
            return corrected_corners
            
        except Exception as e:
            logger.warning(f"Geometry correction failed: {e}")
            return None
    
    def _full_opencv_pipeline(self, image: np.ndarray) -> Optional[List[List[float]]]:
        """Full OpenCV pipeline when YOLO fails"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try chessboard detection on full image
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_size)
            
            if found:
                board_corners = self._extract_board_corners_from_pattern(corners)
                if board_corners is not None:
                    logger.info("✅ Full OpenCV pipeline successful")
                    return board_corners.tolist()
            
            # Fallback: Edge-based detection
            return self._edge_based_detection(image)
            
        except Exception as e:
            logger.warning(f"Full OpenCV pipeline failed: {e}")
            return None
    
    def _edge_based_detection(self, image: np.ndarray) -> Optional[List[List[float]]]:
        """Edge-based corner detection as last resort"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) >= 4:
                # Group and select best lines
                horizontal_lines, vertical_lines = self._group_lines_by_orientation(lines)
                
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    # Select outer lines
                    h_lines = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
                    v_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
                    
                    top_line = h_lines[0]
                    bottom_line = h_lines[-1]
                    left_line = v_lines[0]
                    right_line = v_lines[-1]
                    
                    # Calculate intersections
                    corners = self._calculate_line_intersections([top_line, right_line, bottom_line, left_line])
                    
                    if corners is not None:
                        logger.info("✅ Edge-based detection successful")
                        return corners.tolist()
            
        except Exception as e:
            logger.warning(f"Edge-based detection failed: {e}")
        
        return None
    
    def _group_lines_by_orientation(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Group lines by orientation"""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle > 90:
                angle = 180 - angle
            
            if angle < 30:  # Horizontal
                horizontal_lines.append(line[0])
            elif angle > 60:  # Vertical
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def _calculate_line_intersections(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate intersections of 4 lines"""
        try:
            def line_intersection(line1, line2):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                
                return np.array([x, y])
            
            top_line, right_line, bottom_line, left_line = lines
            
            tl = line_intersection(top_line, left_line)
            tr = line_intersection(top_line, right_line)
            br = line_intersection(bottom_line, right_line)
            bl = line_intersection(bottom_line, left_line)
            
            if all(corner is not None for corner in [tl, tr, br, bl]):
                return np.array([tl, tr, br, bl])
                
        except Exception as e:
            logger.warning(f"Line intersection failed: {e}")
        
        return None
    
    def _log_pipeline_results(self, stages: Dict):
        """Log the results of each pipeline stage"""
        logger.info("📊 Pipeline Stage Results:")
        for stage_name, result in stages.items():
            status = "✅" if result is not None else "❌"
            logger.info(f"   {stage_name.capitalize()}: {status}")
    
    def _visualize_pipeline_stages(self, image: np.ndarray, stages: Dict, image_path: str):
        """Create visualization of all pipeline stages"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Hybrid Corner Detection Pipeline - {Path(image_path).name}', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            stage_names = ['yolo', 'pattern', 'harris', 'subpixel', 'final']
            colors = ['red', 'green', 'blue', 'yellow', 'cyan']
            
            for idx, (stage_name, color) in enumerate(zip(stage_names, colors)):
                if idx >= 5:  # Only 5 subplot positions
                    break
                    
                row = idx // 3
                col = (idx + 1) % 3
                
                axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f'Stage: {stage_name.capitalize()}')
                axes[row, col].axis('off')
                
                corners = stages.get(stage_name)
                if corners is not None:
                    corners_np = np.array(corners)
                    # Draw corners
                    for i, corner in enumerate(corners_np):
                        axes[row, col].plot(corner[0], corner[1], 'o', color=color, markersize=8)
                        axes[row, col].text(corner[0], corner[1], f'{i}', color='white', fontweight='bold')
                    
                    # Draw quadrilateral
                    quad = np.vstack([corners_np, corners_np[0]])  # Close the shape
                    axes[row, col].plot(quad[:, 0], quad[:, 1], '-', color=color, linewidth=2)
            
            plt.tight_layout()
            output_path = f"hybrid_detection_pipeline_{Path(image_path).stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📸 Pipeline visualization saved: {output_path}")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    def comprehensive_evaluation(self, test_images_dir: str, annotations_dir: str, num_samples: int = 50) -> Dict:
        """Comprehensive evaluation against ground truth"""
        logger.info(f"🧪 COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        test_images = list(Path(test_images_dir).glob("*.JPG"))[:num_samples]
        
        results = {
            'total_images': len(test_images),
            'successful_detections': 0,
            'stage_success_rates': {
                'yolo': 0,
                'pattern': 0, 
                'harris': 0,
                'subpixel': 0,
                'final': 0
            },
            'errors': [],
            'processing_times': [],
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
                
                # Test both YOLO-only and hybrid detection
                start_time = time.time()
                
                # YOLO-only baseline
                yolo_corners = self._yolo_detection(str(img_path))
                yolo_error = None
                if yolo_corners:
                    yolo_error = self._calculate_corner_error(gt_corners, yolo_corners)
                
                # Hybrid detection
                hybrid_corners = self.detect_corners_ultra_precise(str(img_path))
                processing_time = time.time() - start_time
                
                if hybrid_corners is not None:
                    results['successful_detections'] += 1
                    
                    # Calculate errors
                    hybrid_error = self._calculate_corner_error(gt_corners, hybrid_corners)
                    results['errors'].append(hybrid_error)
                    results['processing_times'].append(processing_time)
                    
                    # Compare improvement over YOLO
                    if yolo_error is not None:
                        improvement = ((yolo_error - hybrid_error) / yolo_error) * 100
                        results['improvements_over_yolo'].append(improvement)
                        
                        logger.info(f"   {img_path.name}: {hybrid_error:.1f}px (vs YOLO: {yolo_error:.1f}px, {improvement:+.1f}%)")
                    else:
                        logger.info(f"   {img_path.name}: {hybrid_error:.1f}px (YOLO failed)")
                
            except Exception as e:
                logger.warning(f"   Error evaluating {img_path.name}: {e}")
        
        # Calculate summary statistics
        if results['errors']:
            results['average_error'] = np.mean(results['errors'])
            results['median_error'] = np.median(results['errors'])
            results['std_error'] = np.std(results['errors'])
            results['max_error'] = np.max(results['errors'])
            results['min_error'] = np.min(results['errors'])
            results['success_rate'] = (results['successful_detections'] / results['total_images']) * 100
            results['avg_processing_time'] = np.mean(results['processing_times'])
            
            if results['improvements_over_yolo']:
                results['avg_improvement_over_yolo'] = np.mean(results['improvements_over_yolo'])
        
        self._print_evaluation_summary(results)
        return results
    
    def _calculate_corner_error(self, gt_corners: List[List[float]], detected_corners: List[List[float]]) -> float:
        """Calculate average pixel error between ground truth and detected corners"""
        gt_np = np.array(gt_corners)
        det_np = np.array(detected_corners)
        
        # Calculate error for each corner
        errors = np.linalg.norm(gt_np - det_np, axis=1)
        return np.mean(errors)
    
    def _print_evaluation_summary(self, results: Dict):
        """Print comprehensive evaluation summary"""
        logger.info(f"\n🏆 HYBRID CORNER DETECTOR EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 Images processed: {results['total_images']}")
        logger.info(f"✅ Successful detections: {results['successful_detections']}")
        logger.info(f"📈 Success rate: {results.get('success_rate', 0):.1f}%")
        
        if results['errors']:
            logger.info(f"\n🎯 ACCURACY METRICS:")
            logger.info(f"   Average error: {results['average_error']:.1f} pixels")
            logger.info(f"   Median error: {results['median_error']:.1f} pixels")
            logger.info(f"   Error range: {results['min_error']:.1f} - {results['max_error']:.1f} pixels")
            logger.info(f"   Standard deviation: {results['std_error']:.1f} pixels")
            
            logger.info(f"\n⚡ PERFORMANCE:")
            logger.info(f"   Average processing time: {results['avg_processing_time']:.3f} seconds")
            
            if results['improvements_over_yolo']:
                logger.info(f"\n🚀 IMPROVEMENT OVER YOLO:")
                logger.info(f"   Average improvement: {results['avg_improvement_over_yolo']:+.1f}%")

def main():
    """Test the precision corner detector"""
    print("🚀 PRECISION CORNER DETECTION SYSTEM")
    print("=" * 60)
    print("Multi-stage refinement: YOLO → OpenCV → Harris → Sub-pixel → Geometric")
    print()
    
    # Initialize detector with visualization
    detector = PrecisionCornerDetector(enable_visualization=True)
    
    # Test on sample image
    test_image = "my_chess_images/train/images/IMG_4698.JPG"
    if Path(test_image).exists():
        print(f"🧪 Testing on: {test_image}")
        corners = detector.detect_corners_ultra_precise(test_image)
        
        if corners:
            print(f"✅ Detection successful!")
            print(f"   Final corners: {corners}")
        else:
            print("❌ Detection failed")
    
    # Comprehensive evaluation
    test_dirs = [
        ("grey_background_dataset/images/test", "grey_background_dataset/annotations/test"),
        ("grey_background_dataset/images/val", "grey_background_dataset/annotations/val")
    ]
    
    for img_dir, ann_dir in test_dirs:
        if Path(img_dir).exists() and Path(ann_dir).exists():
            print(f"\n📊 Comprehensive evaluation on {img_dir}...")
            results = detector.comprehensive_evaluation(img_dir, ann_dir, num_samples=15)

if __name__ == "__main__":
    main()
