"""Enhanced occupancy detection module with improved accuracy.

This module provides improved methods for detecting chess square occupancy
by using adaptive cropping, multi-scale detection, and better preprocessing.
"""

import numpy as np
import cv2
import chess
from typing import Tuple, List, Optional
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import feature, measure
from sklearn.cluster import KMeans

from chesscog.core import sort_corner_points
from .create_dataset import warp_chessboard_image as original_warp


class EnhancedOccupancyDetector:
    """Enhanced occupancy detector with improved accuracy techniques."""
    
    def __init__(self, square_size: int = 50, margin_ratio: float = 0.3):
        """Initialize the enhanced occupancy detector.
        
        Args:
            square_size: Base size for square detection
            margin_ratio: Ratio of margin to include around each square
        """
        self.square_size = square_size
        self.margin_ratio = margin_ratio
        self.board_size = 8 * square_size
        self.img_size = self.board_size + 2 * square_size
        
    def adaptive_crop_square(self, img: np.ndarray, square: chess.Square, 
                           turn: chess.Color, scale_factor: float = 1.0) -> np.ndarray:
        """Crop a square with adaptive sizing based on position and perspective.
        
        Args:
            img: The warped board image
            square: Chess square to crop
            turn: Current player turn
            scale_factor: Scaling factor for crop size
            
        Returns:
            Cropped square image with adaptive sizing
        """
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if turn == chess.WHITE:
            row, col = 7 - rank, file
        else:
            row, col = rank, 7 - file
        
        # Adaptive margin based on position (larger for distant squares)
        # Squares further from camera appear smaller due to perspective
        perspective_factor = 1.0 + (row / 7.0) * 0.2  # 0-20% increase
        
        # Adaptive width increase based on file position
        if col < 4:
            width_increase = 0.1 + (3 - col) * 0.05  # Left side
        else:
            width_increase = 0.1 + (col - 4) * 0.05  # Right side
        
        # Calculate crop bounds with adaptive margins
        base_margin = self.margin_ratio * perspective_factor * scale_factor
        width_margin = base_margin * (1 + width_increase)
        height_margin = base_margin * perspective_factor
        
        # Calculate crop coordinates
        x1 = int(self.square_size * (col + 0.5 - width_margin))
        x2 = int(self.square_size * (col + 1.5 + width_margin))
        y1 = int(self.square_size * (row + 0.5 - height_margin))
        y2 = int(self.square_size * (row + 1.5 + height_margin))
        
        # Ensure bounds are within image
        x1 = max(0, x1)
        x2 = min(img.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(img.shape[0], y2)
        
        return img[y1:y2, x1:x2]
    
    def multi_scale_crop(self, img: np.ndarray, square: chess.Square, 
                        turn: chess.Color, scales: List[float] = [0.8, 1.0, 1.2]) -> List[np.ndarray]:
        """Crop square at multiple scales for robust detection.
        
        Args:
            img: The warped board image
            square: Chess square to crop
            turn: Current player turn
            scales: List of scale factors to use
            
        Returns:
            List of cropped images at different scales
        """
        crops = []
        for scale in scales:
            crop = self.adaptive_crop_square(img, square, turn, scale)
            if crop.size > 0:
                crops.append(crop)
        return crops
    
    def preprocess_square(self, square_img: np.ndarray, 
                         enhance_contrast: bool = True,
                         denoise: bool = True) -> np.ndarray:
        """Enhanced preprocessing for square images.
        
        Args:
            square_img: Input square image
            enhance_contrast: Whether to enhance contrast
            denoise: Whether to apply denoising
            
        Returns:
            Preprocessed square image
        """
        # Convert to PIL for enhanced processing
        pil_img = Image.fromarray(square_img)
        
        # Enhance contrast
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Apply slight blur to reduce noise
        if denoise:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return np.array(pil_img)
    
    def detect_edges_and_features(self, square_img: np.ndarray) -> Tuple[float, float, float]:
        """Detect edge density and texture features for occupancy detection.
        
        Args:
            square_img: Input square image
            
        Returns:
            Tuple of (edge_density, texture_variance, color_variance)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis using local binary patterns
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        texture_variance = np.var(lbp)
        
        # Color variance
        color_variance = np.var(square_img)
        
        return edge_density, texture_variance, color_variance
    
    def statistical_occupancy_check(self, square_img: np.ndarray, 
                                  empty_threshold: float = 0.15) -> bool:
        """Statistical check for occupancy based on image features.
        
        Args:
            square_img: Input square image
            empty_threshold: Threshold for considering square empty
            
        Returns:
            True if square appears occupied, False otherwise
        """
        edge_density, texture_variance, color_variance = self.detect_edges_and_features(square_img)
        
        # Combine features for occupancy decision
        # Empty squares typically have low edge density and low texture variance
        feature_score = (edge_density * 0.4 + 
                        (texture_variance / 100.0) * 0.3 + 
                        (color_variance / 1000.0) * 0.3)
        
        return feature_score > empty_threshold
    
    def ensemble_occupancy_detection(self, img: np.ndarray, square: chess.Square, 
                                   turn: chess.Color, model, transforms) -> float:
        """Ensemble method combining multiple detection approaches.
        
        Args:
            img: Warped board image
            square: Chess square to analyze
            turn: Current player turn
            model: Trained occupancy model
            transforms: Image transforms for the model
            
        Returns:
            Occupancy probability (0-1)
        """
        # Multi-scale crops
        crops = self.multi_scale_crop(img, square, turn)
        
        if not crops:
            return 0.0
        
        model_scores = []
        statistical_scores = []
        
        for crop in crops:
            if crop.size == 0:
                continue
                
            # Preprocess crop
            processed_crop = self.preprocess_square(crop)
            
            # Statistical analysis
            stat_occupied = self.statistical_occupancy_check(processed_crop)
            statistical_scores.append(float(stat_occupied))
            
            # Model prediction
            try:
                # Resize to model input size
                resized_crop = cv2.resize(processed_crop, (100, 100))
                pil_crop = Image.fromarray(resized_crop)
                tensor_crop = transforms(pil_crop).unsqueeze(0)
                
                with torch.no_grad():
                    model_output = model(tensor_crop)
                    prob = F.softmax(model_output, dim=1)[0, 1].item()  # Probability of occupied
                    model_scores.append(prob)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                continue
        
        # Combine scores
        if model_scores:
            avg_model_score = np.mean(model_scores)
        else:
            avg_model_score = 0.0
            
        if statistical_scores:
            avg_stat_score = np.mean(statistical_scores)
        else:
            avg_stat_score = 0.0
        
        # Weighted combination (favor model if available)
        if model_scores:
            final_score = 0.7 * avg_model_score + 0.3 * avg_stat_score
        else:
            final_score = avg_stat_score
            
        return final_score
    
    def enhanced_warp_chessboard(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Enhanced warping with perspective correction and quality improvements.
        
        Args:
            img: Input image
            corners: Corner coordinates
            
        Returns:
            Warped and enhanced board image
        """
        # First apply standard warping
        warped = original_warp(img, corners)
        
        # Apply additional enhancement
        warped = self.enhance_warped_image(warped)
        
        return warped
    
    def enhance_warped_image(self, warped_img: np.ndarray) -> np.ndarray:
        """Apply enhancements to warped board image.
        
        Args:
            warped_img: Warped board image
            
        Returns:
            Enhanced warped image
        """
        # Convert to PIL for better processing
        pil_img = Image.fromarray(warped_img)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Enhance brightness slightly
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.05)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Apply slight denoising
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return np.array(pil_img)
    
    def confidence_based_filtering(self, occupancy_probs: np.ndarray, 
                                 confidence_threshold: float = 0.6) -> np.ndarray:
        """Apply confidence-based filtering to occupancy predictions.
        
        Args:
            occupancy_probs: Array of occupancy probabilities
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Filtered occupancy predictions
        """
        # Convert probabilities to binary with confidence filtering
        confident_occupied = occupancy_probs > confidence_threshold
        confident_empty = occupancy_probs < (1 - confidence_threshold)
        
        # For uncertain squares, use additional heuristics
        uncertain_mask = ~(confident_occupied | confident_empty)
        
        # Apply chess-specific heuristics for uncertain squares
        result = np.zeros_like(occupancy_probs, dtype=bool)
        result[confident_occupied] = True
        
        # For uncertain squares, be more conservative (slightly favor empty)
        uncertain_threshold = 0.55  # Slightly higher threshold for uncertain squares
        result[uncertain_mask] = occupancy_probs[uncertain_mask] > uncertain_threshold
        
        return result