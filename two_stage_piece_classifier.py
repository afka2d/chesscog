#!/usr/bin/env python3
"""
Two-Stage Piece Classifier
Stage 1: Color classification (white/black)
Stage 2: Piece type classification (pawn, rook, knight, bishop, queen, king)

This approach eliminates color confusion and improves overall accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ColorClassifier(nn.Module):
    """Stage 1: Classify piece color (white/black)"""
    
    def __init__(self):
        super().__init__()
        # Simple CNN for color classification
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 12 * 25, 256)  # For 100x200 input
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: white, black
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 25)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PieceTypeClassifier(nn.Module):
    """Stage 2: Classify piece type (pawn, rook, knight, bishop, queen, king)"""
    
    def __init__(self):
        super().__init__()
        # CNN for piece type classification
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256 * 6 * 12, 512)  # For 100x200 input
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 6)  # 6 piece types
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 6 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TwoStagePieceClassifier:
    """Main two-stage piece classifier"""
    
    def __init__(self, models_dir="two_stage_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.color_classifier = ColorClassifier()
        self.piece_type_classifier = PieceTypeClassifier()
        
        # Define transforms (matching your current setup)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 200)),  # Match your current transform size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class mappings
        self.color_classes = ['white', 'black']
        self.piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        # Load pre-trained models if available
        self._load_models()
        
        # Set models to evaluation mode
        self.color_classifier.eval()
        self.piece_type_classifier.eval()
        
        logger.info("Two-stage piece classifier initialized")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            # Load color classifier
            color_model_path = self.models_dir / "color_classifier.pt"
            if color_model_path.exists():
                self.color_classifier.load_state_dict(torch.load(color_model_path, map_location='cpu'))
                logger.info("Loaded pre-trained color classifier")
            
            # Load piece type classifier
            piece_model_path = self.models_dir / "piece_type_classifier.pt"
            if piece_model_path.exists():
                self.piece_type_classifier.load_state_dict(torch.load(piece_model_path, map_location='cpu'))
                logger.info("Loaded pre-trained piece type classifier")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Using randomly initialized models")
    
    def classify_piece(self, square_img, confidence_threshold=0.6):
        """
        Classify a single square image using two-stage approach.
        
        Args:
            square_img: numpy array of the square image
            confidence_threshold: minimum confidence for predictions
            
        Returns:
            tuple: (piece_name, confidence, stage1_color_conf, stage2_piece_conf)
        """
        try:
            # Convert to tensor
            input_tensor = self.transform(square_img).unsqueeze(0)
            
            # Stage 1: Color classification
            with torch.no_grad():
                color_output = self.color_classifier(input_tensor)
                color_probs = F.softmax(color_output, dim=1)
                color_pred = torch.argmax(color_probs, dim=1).item()
                color_conf = color_probs[0][color_pred].item()
                
                # Stage 2: Piece type classification
                piece_output = self.piece_type_classifier(input_tensor)
                piece_probs = F.softmax(piece_output, dim=1)
                piece_pred = torch.argmax(piece_probs, dim=1).item()
                piece_conf = piece_probs[0][piece_pred].item()
            
            # Check confidence thresholds
            if color_conf < confidence_threshold or piece_conf < confidence_threshold:
                return None, 0.0, color_conf, piece_conf
            
            # Combine predictions
            color_name = self.color_classes[color_pred]
            piece_name = self.piece_types[piece_pred]
            full_piece_name = f"{color_name}_{piece_name}"
            
            # Overall confidence is average of both stages
            overall_confidence = (color_conf + piece_conf) / 2
            
            return full_piece_name, overall_confidence, color_conf, piece_conf
            
        except Exception as e:
            logger.error(f"Error in two-stage classification: {e}")
            return None, 0.0, 0.0, 0.0
    
    def classify_board(self, img, corners, occupancy):
        """
        Classify all pieces on the chess board.
        
        Args:
            img: full chess board image
            corners: corner coordinates
            occupancy: occupancy map (8x8 boolean array)
            
        Returns:
            numpy array: 8x8 array with piece objects or None
        """
        try:
            from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image, crop_square
            
            # Warp the chessboard
            warped = warp_chessboard_image(img, corners)
            
            # Initialize pieces array
            pieces = np.full((8, 8), None, dtype=object)
            
            # Process each occupied square
            for rank in range(8):
                for file in range(8):
                    if occupancy[rank, file]:
                        # Crop the square
                        square_img = crop_square(warped, rank, file)
                        
                        # Classify the piece
                        piece_name, confidence, color_conf, piece_conf = self.classify_piece(square_img)
                        
                        if piece_name and confidence > 0.5:
                            # Convert to chess piece
                            piece_obj = self._name_to_chess_piece(piece_name)
                            if piece_obj:
                                pieces[rank, file] = piece_obj
                                logger.debug(f"Square {rank},{file}: {piece_name} (conf: {confidence:.3f})")
                            else:
                                logger.warning(f"Could not convert {piece_name} to chess piece")
                        else:
                            logger.debug(f"Square {rank},{file}: Low confidence (color: {color_conf:.3f}, piece: {piece_conf:.3f})")
            
            return pieces
            
        except Exception as e:
            logger.error(f"Error in board classification: {e}")
            return np.full((8, 8), None, dtype=object)
    
    def _name_to_chess_piece(self, piece_name):
        """Convert piece name to chess.Piece object"""
        try:
            import chess
            
            # Parse piece name (e.g., "white_pawn" -> chess.WHITE, chess.PAWN)
            if piece_name.startswith('white_'):
                color = chess.WHITE
                piece_type = piece_name[6:]  # Remove 'white_' prefix
            else:
                color = chess.BLACK
                piece_type = piece_name[6:]  # Remove 'black_' prefix
            
            # Map piece type to chess constant
            piece_map = {
                'pawn': chess.PAWN,
                'rook': chess.ROOK,
                'knight': chess.KNIGHT,
                'bishop': chess.BISHOP,
                'queen': chess.QUEEN,
                'king': chess.KING
            }
            
            if piece_type in piece_map:
                return chess.Piece(piece_map[piece_type], color)
            else:
                logger.warning(f"Unknown piece type: {piece_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting piece name: {e}")
            return None
    
    def save_models(self):
        """Save the trained models"""
        try:
            torch.save(self.color_classifier.state_dict(), self.models_dir / "color_classifier.pt")
            torch.save(self.piece_type_classifier.state_dict(), self.models_dir / "piece_type_classifier.pt")
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_model_info(self):
        """Get information about the models"""
        return {
            "color_classifier_params": sum(p.numel() for p in self.color_classifier.parameters()),
            "piece_type_classifier_params": sum(p.numel() for p in self.piece_type_classifier.parameters()),
            "transform_size": (100, 200),
            "color_classes": self.color_classes,
            "piece_types": self.piece_types,
            "total_classes": len(self.color_classes) * len(self.piece_types)
        }

# Test function
def test_two_stage_classifier():
    """Test the two-stage classifier"""
    print("🧪 Testing Two-Stage Piece Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = TwoStagePieceClassifier()
    
    # Get model info
    info = classifier.get_model_info()
    print(f"📊 Model Information:")
    print(f"  Color classifier parameters: {info['color_classifier_params']:,}")
    print(f"  Piece type classifier parameters: {info['piece_type_classifier_params']:,}")
    print(f"  Transform size: {info['transform_size']}")
    print(f"  Total possible classes: {info['total_classes']}")
    
    # Test with a sample image
    test_dir = Path("grey_background_dataset/pieces/test")
    if test_dir.exists():
        # Find a test image
        test_images = list(test_dir.glob("*/**/*.png"))
        if test_images:
            test_img_path = test_images[0]
            print(f"\n🔍 Testing with: {test_img_path.name}")
            
            # Load and test image
            img = cv2.imread(str(test_img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Classify
                piece_name, confidence, color_conf, piece_conf = classifier.classify_piece(img)
                
                print(f"  Result: {piece_name}")
                print(f"  Overall confidence: {confidence:.3f}")
                print(f"  Color confidence: {color_conf:.3f}")
                print(f"  Piece type confidence: {piece_conf:.3f}")
            else:
                print("  ❌ Could not load test image")
        else:
            print("  ⚠️  No test images found")
    else:
        print("  ⚠️  Test directory not found")
    
    print("\n✅ Two-stage classifier test completed")

if __name__ == "__main__":
    test_two_stage_classifier()
