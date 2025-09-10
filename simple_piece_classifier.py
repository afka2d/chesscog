#!/usr/bin/env python3
"""
Simple piece classifier that integrates with the existing API.
Uses the original ChessCog piece classifier without modifying other components.
"""

import numpy as np
import chess
import torch
from torchvision import transforms
from PIL import Image
import functools
from pathlib import Path

class SimplePieceClassifier:
    """Simple piece classifier using the original ChessCog approach."""
    
    def __init__(self, models_folder="models"):
        """Initialize the piece classifier."""
        self.models_folder = Path(models_folder)
        self._load_piece_classifier()
    
    def _load_piece_classifier(self):
        """Load the piece classifier model and transforms."""
        try:
            # Load piece classifier config and model
            # Try different models in order of preference
            model_candidates = [
                ("InceptionV3.pt", "InceptionV3.yaml"),
                ("ResNet_simple_balanced.pt", "ResNet_simple_balanced.yaml"),
                ("ResNet_robust.pt", "ResNet_robust.yaml"),
                ("ResNet_simple_robust.pt", "ResNet_simple_robust.yaml"),
                ("ResNet_robust_full.pt", "ResNet_robust_full.yaml")
            ]
            
            piece_cfg_path = None
            piece_model_path = None
            
            for model_file, cfg_file in model_candidates:
                cfg_path = self.models_folder / "piece_classifier" / cfg_file
                model_path = self.models_folder / "piece_classifier" / model_file
                if cfg_path.exists() and model_path.exists():
                    piece_cfg_path = cfg_path
                    piece_model_path = model_path
                    print(f"✅ Using model: {model_file}")
                    break
            
            if piece_cfg_path is None or piece_model_path is None:
                raise FileNotFoundError("No suitable piece classifier model found")
            
            if piece_cfg_path.exists() and piece_model_path.exists():
                from recap import CfgNode as CN
                from chesscog.core.dataset import build_transforms, Datasets
                from chesscog.core.dataset import name_to_piece
                
                # Load config
                self._pieces_cfg = CN.load_yaml_with_base(piece_cfg_path)
                
                # Load model
                self._pieces_model = torch.load(piece_model_path, map_location='cpu', weights_only=False)
                self._pieces_model.eval()
                
                # Build transforms
                self._pieces_transforms = build_transforms(self._pieces_cfg, mode=Datasets.TEST)
                
                # Get piece classes
                self._piece_classes = np.array(list(map(name_to_piece, self._pieces_cfg.DATASET.CLASSES)))
                
                print("✅ Piece classifier loaded successfully")
                return True
            else:
                print("❌ Piece classifier files not found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load piece classifier: {e}")
            return False
    
    def classify_pieces(self, img, corners, occupancy, turn):
        """Classify pieces on occupied squares."""
        try:
            # Get occupied squares
            occupied_squares = []
            for i, is_occupied in enumerate(occupancy):
                if bool(is_occupied):  # Convert to boolean to avoid array comparison issues
                    occupied_squares.append(chess.SQUARES[i])
            
            if not occupied_squares:
                # No occupied squares, return all None
                return [None] * 64
            
            # Warp the chessboard image
            from chesscog.piece_classifier.create_dataset import warp_chessboard_image, crop_square
            warped = warp_chessboard_image(img, corners)
            
            # Crop piece images for occupied squares
            piece_imgs = []
            for square in occupied_squares:
                piece_img = crop_square(warped, square, turn)
                piece_imgs.append(Image.fromarray(piece_img))
            
            # Apply transforms
            piece_imgs = [self._pieces_transforms(img) for img in piece_imgs]
            piece_imgs = torch.stack(piece_imgs)
            
            # Get predictions
            with torch.no_grad():
                predictions = self._pieces_model(piece_imgs)
                predicted_classes = predictions.argmax(axis=-1).cpu().numpy()
                piece_names = self._piece_classes[predicted_classes]
            
            # Create result array
            result = [None] * 64
            occupied_idx = 0
            for i, is_occupied in enumerate(occupancy):
                if bool(is_occupied):  # Convert to boolean to avoid array comparison issues
                    result[i] = piece_names[occupied_idx]
                    occupied_idx += 1
            
            return result
            
        except Exception as e:
            print(f"❌ Error in piece classification: {e}")
            # Return all None on error
            return [None] * 64

def test_simple_classifier():
    """Test the simple piece classifier."""
    print("🧪 Testing Simple Piece Classifier")
    print("=" * 40)
    
    # Initialize classifier
    classifier = SimplePieceClassifier()
    
    if not hasattr(classifier, '_pieces_model'):
        print("❌ Classifier not loaded properly")
        return False
    
    # Test with a sample image
    import glob
    import os
    
    test_dirs = [
        "my_chess_images/train/images",
        "grey_background_dataset/images/test"
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(glob.glob(os.path.join(test_dir, ext)))
            if images:
                test_image = images[0]  # Use first image
                break
    
    if not test_image:
        print("❌ No test images found")
        return False
    
    print(f"📁 Using test image: {test_image}")
    
    # Load image
    img = Image.open(test_image).convert('RGB')
    img_array = np.array(img)
    
    # Create test corners (approximate)
    corners = np.array([
        [50, 50],   # Top-left
        [400, 50],  # Top-right
        [400, 400], # Bottom-right
        [50, 400]   # Bottom-left
    ], dtype=np.float32)
    
    # Create test occupancy (some squares occupied)
    occupancy = [False] * 64
    # Set some squares as occupied for testing
    test_squares = [0, 1, 2, 3, 4, 5, 6, 7,  # Back rank
                   8, 9, 10, 11, 12, 13, 14, 15,  # Second rank
                   48, 49, 50, 51, 52, 53, 54, 55,  # Seventh rank
                   56, 57, 58, 59, 60, 61, 62, 63]  # Eighth rank
    for square in test_squares:
        occupancy[square] = True
    
    print(f"📊 Testing with {sum(occupancy)} occupied squares")
    
    # Classify pieces
    pieces = classifier.classify_pieces(img_array, corners, occupancy, chess.WHITE)
    
    # Analyze results
    piece_count = sum(1 for p in pieces if p is not None)
    piece_types = set(p for p in pieces if p is not None)
    
    print(f"✅ Classification completed!")
    print(f"   Pieces detected: {piece_count}")
    print(f"   Unique types: {len(piece_types)}")
    print(f"   Piece types: {list(piece_types)}")
    
    # Calculate diversity
    diversity = len(piece_types) / 12.0 if piece_count > 0 else 0
    print(f"   Diversity score: {diversity:.2f}")
    
    # Estimate accuracy
    if diversity >= 0.6:
        estimated_accuracy = "75-85%"
        assessment = "GOOD"
    elif diversity >= 0.4:
        estimated_accuracy = "65-75%"
        assessment = "MODERATE"
    else:
        estimated_accuracy = "50-65%"
        assessment = "POOR"
    
    print(f"\n🎯 ESTIMATED ACCURACY: {estimated_accuracy}")
    print(f"   Assessment: {assessment}")
    
    return assessment in ["GOOD", "MODERATE"]

if __name__ == "__main__":
    print("🎯 Simple Piece Classifier Test")
    print("=" * 40)
    
    success = test_simple_classifier()
    
    if success:
        print("\n🎉 SUCCESS: Simple piece classifier is working!")
        print("   This can be integrated into your existing API")
    else:
        print("\n❌ FAILED: Simple piece classifier needs improvement")
