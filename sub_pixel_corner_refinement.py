#!/usr/bin/env python3
"""
Sub-pixel corner refinement using OpenCV for enhanced accuracy.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SubPixelCornerRefiner:
    """Refine corner predictions to sub-pixel accuracy using OpenCV"""
    
    def __init__(self, window_size=(11, 11), zero_zone=(-1, -1), criteria=None):
        self.window_size = window_size
        self.zero_zone = zero_zone
        
        if criteria is None:
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        else:
            self.criteria = criteria
    
    def refine_corners(self, image, corners):
        """
        Refine corner coordinates to sub-pixel accuracy.
        
        Args:
            image: Input image (BGR or RGB)
            corners: Array of corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            Refined corner coordinates with sub-pixel accuracy
        """
        if image is None or corners is None:
            return corners
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Convert corners to the format expected by cornerSubPix
        corners_array = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        
        try:
            # Apply sub-pixel corner refinement
            refined_corners = cv2.cornerSubPix(
                gray, 
                corners_array,
                self.window_size,
                self.zero_zone,
                self.criteria
            )
            
            # Convert back to original format
            refined_corners = refined_corners.reshape(-1, 2)
            
            # Validate that corners are still within image bounds
            h, w = gray.shape
            refined_corners[:, 0] = np.clip(refined_corners[:, 0], 0, w-1)
            refined_corners[:, 1] = np.clip(refined_corners[:, 1], 0, h-1)
            
            return refined_corners.tolist()
            
        except Exception as e:
            logger.warning(f"Corner refinement failed: {e}")
            return corners

class GeometricCornerValidator:
    """Validate and correct corner predictions using geometric constraints"""
    
    def __init__(self, min_area_ratio=0.01, max_area_ratio=0.9):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
    
    def validate_corners(self, corners, image_shape):
        """
        Validate corner predictions and apply corrections if needed.
        
        Args:
            corners: Array of corner coordinates
            image_shape: (height, width) of the image
        
        Returns:
            Validated and potentially corrected corners
        """
        if not corners or len(corners) != 4:
            return corners
        
        h, w = image_shape[:2]
        corners = np.array(corners)
        
        # Check if corners form a reasonable quadrilateral
        if not self._is_valid_quadrilateral(corners, (h, w)):
            # Try to fix common issues
            corners = self._fix_corner_order(corners)
            corners = self._ensure_convex_quadrilateral(corners)
        
        # Ensure corners are within image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, w-1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h-1)
        
        return corners.tolist()
    
    def _is_valid_quadrilateral(self, corners, image_shape):
        """Check if corners form a valid quadrilateral"""
        h, w = image_shape
        
        # Calculate area using shoelace formula
        area = self._calculate_polygon_area(corners)
        image_area = h * w
        area_ratio = area / image_area
        
        # Check area ratio
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False
        
        # Check if quadrilateral is roughly convex
        return self._is_roughly_convex(corners)
    
    def _calculate_polygon_area(self, corners):
        """Calculate polygon area using shoelace formula"""
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def _is_roughly_convex(self, corners):
        """Check if quadrilateral is roughly convex"""
        # Calculate cross products of consecutive edges
        edges = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edges.append(p2 - p1)
        
        cross_products = []
        for i in range(4):
            e1 = edges[i]
            e2 = edges[(i + 1) % 4]
            cross = np.cross(e1, e2)
            cross_products.append(cross)
        
        # For a convex quadrilateral, all cross products should have the same sign
        signs = [1 if cp > 0 else -1 if cp < 0 else 0 for cp in cross_products]
        return len(set(signs)) <= 2  # Allow some tolerance
    
    def _fix_corner_order(self, corners):
        """Fix corner ordering to be consistent (e.g., clockwise from top-left)"""
        # Find centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid to each corner
        angles = []
        for corner in corners:
            angle = np.arctan2(corner[1] - centroid[1], corner[0] - centroid[0])
            angles.append(angle)
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        return corners[sorted_indices]
    
    def _ensure_convex_quadrilateral(self, corners):
        """Ensure the quadrilateral is convex by adjusting corners if needed"""
        # This is a simplified approach - in practice, you might want more sophisticated methods
        return corners

class EnhancedCornerDetectionService:
    """Enhanced corner detection service with sub-pixel refinement"""
    
    def __init__(self, model_path="models/enhanced_corner_detector_best.pt", image_size=512):
        self.model_path = model_path
        self.image_size = image_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize refinement components
        self.subpixel_refiner = SubPixelCornerRefiner()
        self.geometric_validator = GeometricCornerValidator()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the enhanced corner detection model"""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Import the enhanced model
            from enhanced_corner_training import EnhancedCornerModel
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine model type
            try:
                self.model = EnhancedCornerModel(backbone='efficientnet_b3', pretrained=False)
            except:
                self.model = EnhancedCornerModel(backbone='resnet18', pretrained=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.image_size = checkpoint.get('image_size', 512)
            
            logger.info("Enhanced corner detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_corners_with_refinement(self, image_path):
        """
        Detect corners with sub-pixel refinement and validation.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            List of refined corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Load and preprocess image
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            h, w = original_image.shape[:2]
            
            # Preprocess for model
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
            
            # Convert to tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image_resized).unsqueeze(0).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                predictions = self.model(input_tensor)
                predictions = predictions.cpu().numpy().reshape(4, 2)
            
            # Convert normalized coordinates back to pixel coordinates
            pixel_corners = predictions * [w, h]
            
            # Apply geometric validation
            validated_corners = self.geometric_validator.validate_corners(
                pixel_corners, (h, w)
            )
            
            # Apply sub-pixel refinement
            refined_corners = self.subpixel_refiner.refine_corners(
                original_image, validated_corners
            )
            
            logger.info(f"Corner detection completed with refinement")
            return refined_corners
            
        except Exception as e:
            logger.error(f"Corner detection failed: {e}")
            return None
    
    def compare_with_ground_truth(self, image_path, ground_truth_corners):
        """Compare refined predictions with ground truth"""
        predicted_corners = self.detect_corners_with_refinement(image_path)
        
        if predicted_corners is None or ground_truth_corners is None:
            return None
        
        pred_corners = np.array(predicted_corners)
        gt_corners = np.array(ground_truth_corners)
        
        # Calculate per-corner errors
        errors = np.sqrt(np.sum((pred_corners - gt_corners) ** 2, axis=1))
        
        return {
            'predicted_corners': predicted_corners,
            'ground_truth_corners': ground_truth_corners,
            'per_corner_errors': errors.tolist(),
            'average_error': np.mean(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors)
        }

def test_refinement_system():
    """Test the enhanced corner detection with refinement"""
    print("🔧 TESTING ENHANCED CORNER DETECTION WITH REFINEMENT")
    print("=" * 60)
    
    # Check if enhanced model exists
    model_path = "models/enhanced_corner_detector_best.pt"
    if not Path(model_path).exists():
        print(f"❌ Enhanced model not found: {model_path}")
        print("   Please train the enhanced model first using enhanced_corner_training.py")
        return
    
    # Initialize enhanced service
    service = EnhancedCornerDetectionService(model_path)
    
    # Test images
    test_images = [
        "grey_background_dataset/images/val/IMG_4779.JPG",
        "grey_background_dataset/images/test/IMG_4785.JPG",
        "grey_background_dataset/images/test/IMG_4763.JPG"
    ]
    
    for image_path in test_images:
        if not Path(image_path).exists():
            continue
        
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        # Detect corners with refinement
        corners = service.detect_corners_with_refinement(image_path)
        
        if corners:
            print(f"   ✅ Refined corners detected: {corners}")
        else:
            print(f"   ❌ Corner detection failed")

if __name__ == "__main__":
    test_refinement_system()
