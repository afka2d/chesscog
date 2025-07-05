"""Enhanced chess recognition module with improved occupancy detection.

This module extends the original ChessRecognizer with enhanced occupancy detection
techniques for better accuracy when working with corrected board corners.
"""

import numpy as np
import chess
import functools
import cv2
import torch
from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image

from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.occupancy_classifier.enhanced_detection import EnhancedOccupancyDetector
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from chesscog.core import device, DEVICE
from chesscog.core.dataset import build_transforms, Datasets
from chesscog.core.dataset import name_to_piece
from chesscog.recognition.recognition import ChessRecognizer
from recap import URI, CfgNode as CN


class EnhancedChessRecognizer(ChessRecognizer):
    """Enhanced chess recognizer with improved occupancy detection.
    
    This class extends the original ChessRecognizer with advanced occupancy
    detection techniques including adaptive cropping, multi-scale detection,
    and ensemble methods.
    """
    
    def __init__(self, classifiers_folder: Path = URI("models://"), 
                 use_enhanced_detection: bool = True,
                 confidence_threshold: float = 0.6):
        """Initialize the enhanced chess recognizer.
        
        Args:
            classifiers_folder: Path to classifier models
            use_enhanced_detection: Whether to use enhanced occupancy detection
            confidence_threshold: Confidence threshold for predictions
        """
        super().__init__(classifiers_folder)
        self.use_enhanced_detection = use_enhanced_detection
        self.confidence_threshold = confidence_threshold
        
        if use_enhanced_detection:
            self.enhanced_detector = EnhancedOccupancyDetector()
    
    def _classify_occupancy_enhanced(self, img: np.ndarray, turn: chess.Color, 
                                   corners: np.ndarray) -> np.ndarray:
        """Enhanced occupancy classification with improved techniques.
        
        Args:
            img: Input image
            turn: Current player turn
            corners: Board corner coordinates
            
        Returns:
            Array of occupancy probabilities for each square
        """
        # Use enhanced warping if available
        if hasattr(self.enhanced_detector, 'enhanced_warp_chessboard'):
            warped = self.enhanced_detector.enhanced_warp_chessboard(img, corners)
        else:
            warped = create_occupancy_dataset.warp_chessboard_image(img, corners)
        
        occupancy_probs = []
        
        for square in self._squares:
            # Use ensemble detection for each square
            prob = self.enhanced_detector.ensemble_occupancy_detection(
                warped, square, turn, self._occupancy_model, self._occupancy_transforms
            )
            occupancy_probs.append(prob)
        
        occupancy_probs = np.array(occupancy_probs)
        
        # Apply confidence-based filtering
        occupancy_binary = self.enhanced_detector.confidence_based_filtering(
            occupancy_probs, self.confidence_threshold
        )
        
        return occupancy_binary
    
    def _classify_occupancy_with_fallback(self, img: np.ndarray, turn: chess.Color, 
                                        corners: np.ndarray) -> np.ndarray:
        """Occupancy classification with fallback to original method.
        
        Args:
            img: Input image
            turn: Current player turn
            corners: Board corner coordinates
            
        Returns:
            Array of occupancy predictions for each square
        """
        try:
            if self.use_enhanced_detection:
                return self._classify_occupancy_enhanced(img, turn, corners)
            else:
                return self._classify_occupancy(img, turn, corners)
        except Exception as e:
            print(f"Enhanced detection failed, falling back to original: {e}")
            return self._classify_occupancy(img, turn, corners)
    
    def _validate_occupancy_consistency(self, occupancy: np.ndarray, 
                                      pieces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate consistency between occupancy and piece predictions.
        
        Args:
            occupancy: Occupancy predictions
            pieces: Piece predictions
            
        Returns:
            Tuple of corrected (occupancy, pieces) arrays
        """
        # Ensure piece predictions are consistent with occupancy
        corrected_pieces = pieces.copy()
        corrected_occupancy = occupancy.copy()
        
        # If a square has a piece but is marked as empty, mark it as occupied
        has_piece = pieces != None
        corrected_occupancy[has_piece] = True
        
        # If a square is marked as occupied but has no piece, set piece to None
        corrected_pieces[~corrected_occupancy] = None
        
        return corrected_occupancy, corrected_pieces
    
    def predict_with_enhanced_occupancy(self, img: np.ndarray, 
                                      turn: chess.Color = chess.WHITE,
                                      corners: Optional[np.ndarray] = None) -> Tuple[chess.Board, np.ndarray]:
        """Perform chess recognition with enhanced occupancy detection.
        
        Args:
            img: Input image (RGB)
            turn: Current player turn
            corners: Optional pre-computed corners
            
        Returns:
            Tuple of (predicted board, corner coordinates)
        """
        with torch.no_grad():
            # Handle corner detection
            if corners is None:
                img_resized, img_scale = resize_image(self._corner_detection_cfg, img)
                corners, _ = find_corners(self._corner_detection_cfg, img_resized)
                corners = corners / img_scale
                img_for_classification = img
            else:
                img_for_classification = img
            
            # Enhanced occupancy classification
            occupancy = self._classify_occupancy_with_fallback(img_for_classification, turn, corners)
            
            # Piece classification
            pieces = self._classify_pieces(img_for_classification, turn, corners, occupancy)
            
            # Validate consistency
            occupancy, pieces = self._validate_occupancy_consistency(occupancy, pieces)
            
            # Build chess board
            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            
            return board, corners
    
    def predict_with_debug_enhanced(self, img: np.ndarray, 
                                  turn: chess.Color = chess.WHITE,
                                  corners: Optional[np.ndarray] = None) -> Tuple[chess.Board, np.ndarray, dict]:
        """Perform enhanced recognition with detailed debug information.
        
        Args:
            img: Input image (RGB)
            turn: Current player turn
            corners: Optional pre-computed corners
            
        Returns:
            Tuple of (predicted board, corners, debug images dict)
        """
        with torch.no_grad():
            debug_images = {}
            
            # Handle corner detection
            if corners is None:
                img_resized, img_scale = resize_image(self._corner_detection_cfg, img)
                corners_temp, corner_debug = find_corners(self._corner_detection_cfg, img_resized)
                debug_images.update(corner_debug)
                corners = corners_temp / img_scale
                img_for_classification = img
            else:
                img_for_classification = img
                debug_images['corners_provided'] = self._visualize_corners(img, corners)
            
            # Enhanced warped board
            if self.use_enhanced_detection:
                warped_board = self.enhanced_detector.enhanced_warp_chessboard(img_for_classification, corners)
                debug_images['enhanced_warped_board'] = warped_board.copy()
            else:
                warped_board = create_occupancy_dataset.warp_chessboard_image(img_for_classification, corners)
                debug_images['warped_board'] = warped_board.copy()
            
            # Enhanced occupancy classification
            occupancy = self._classify_occupancy_with_fallback(img_for_classification, turn, corners)
            
            # Create enhanced occupancy visualization
            debug_images['enhanced_occupancy_map'] = self._visualize_enhanced_occupancy(
                warped_board, occupancy, turn
            )
            
            # Piece classification
            pieces = self._classify_pieces(img_for_classification, turn, corners, occupancy)
            
            # Validate consistency
            occupancy, pieces = self._validate_occupancy_consistency(occupancy, pieces)
            
            # Enhanced piece visualization
            debug_images['enhanced_piece_map'] = self._visualize_piece_map(
                warped_board, pieces, occupancy, turn
            )
            
            # Build chess board
            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            
            return board, corners, debug_images
    
    def _visualize_corners(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Visualize corner points on the image.
        
        Args:
            img: Input image
            corners: Corner coordinates
            
        Returns:
            Image with corner points visualized
        """
        vis_img = img.copy()
        for i, corner in enumerate(corners):
            cv2.circle(vis_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), tuple(corner.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return vis_img
    
    def _visualize_enhanced_occupancy(self, warped_board: np.ndarray, 
                                    occupancy: np.ndarray, turn: chess.Color) -> np.ndarray:
        """Enhanced visualization of occupancy results.
        
        Args:
            warped_board: Warped board image
            occupancy: Occupancy predictions
            turn: Current player turn
            
        Returns:
            Visualized occupancy map
        """
        vis = warped_board.copy()
        square_size = vis.shape[0] // 10
        
        for idx, occupied in enumerate(occupancy):
            rank = chess.square_rank(self._squares[idx])
            file = chess.square_file(self._squares[idx])
            
            if turn == chess.WHITE:
                row, col = 7 - rank, file
            else:
                row, col = rank, 7 - file
            
            # Calculate square bounds
            x1 = int((col + 1) * square_size)
            y1 = int((row + 1) * square_size)
            x2 = int((col + 2) * square_size)
            y2 = int((row + 2) * square_size)
            
            # Draw enhanced visualization
            if occupied:
                # Green for occupied
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Add filled circle in center
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(vis, center, 8, (0, 255, 0), -1)
            else:
                # Red for empty
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Add X for empty
                cv2.line(vis, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (255, 0, 0), 2)
                cv2.line(vis, (x1 + 5, y2 - 5), (x2 - 5, y1 + 5), (255, 0, 0), 2)
        
        return vis
    
    def get_occupancy_statistics(self, img: np.ndarray, turn: chess.Color, 
                               corners: np.ndarray) -> dict:
        """Get detailed statistics about occupancy detection.
        
        Args:
            img: Input image
            turn: Current player turn
            corners: Board corner coordinates
            
        Returns:
            Dictionary with occupancy statistics
        """
        if not self.use_enhanced_detection:
            return {}
        
        warped = self.enhanced_detector.enhanced_warp_chessboard(img, corners)
        stats = {
            'square_features': [],
            'occupancy_confidence': [],
            'overall_stats': {}
        }
        
        total_confident = 0
        total_uncertain = 0
        
        for square in self._squares:
            # Get multi-scale crops
            crops = self.enhanced_detector.multi_scale_crop(warped, square, turn)
            
            square_stats = {
                'square': chess.square_name(square),
                'num_crops': len(crops),
                'features': []
            }
            
            for crop in crops:
                if crop.size > 0:
                    processed_crop = self.enhanced_detector.preprocess_square(crop)
                    edge_density, texture_var, color_var = self.enhanced_detector.detect_edges_and_features(processed_crop)
                    square_stats['features'].append({
                        'edge_density': edge_density,
                        'texture_variance': texture_var,
                        'color_variance': color_var
                    })
            
            # Get occupancy probability
            prob = self.enhanced_detector.ensemble_occupancy_detection(
                warped, square, turn, self._occupancy_model, self._occupancy_transforms
            )
            
            square_stats['occupancy_prob'] = prob
            
            if prob > 0.6 or prob < 0.4:
                total_confident += 1
            else:
                total_uncertain += 1
            
            stats['square_features'].append(square_stats)
            stats['occupancy_confidence'].append(prob)
        
        stats['overall_stats'] = {
            'confident_predictions': total_confident,
            'uncertain_predictions': total_uncertain,
            'confidence_ratio': total_confident / 64,
            'avg_confidence': np.mean([abs(p - 0.5) for p in stats['occupancy_confidence']])
        }
        
        return stats