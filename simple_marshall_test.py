#!/usr/bin/env python3
"""
Simple test of Marshall occupancy model
Test the model directly on unseen Marshall data to check for overfitting
"""

import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
from sklearn.model_selection import train_test_split

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_square_for_occupancy(square):
    """Preprocess square for occupancy detection"""
    square = cv2.resize(square, (100, 100))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    square = square.astype(np.float32) / 255.0
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def warp_board(image, corners):
    """Warp image to get a square chessboard"""
    try:
        src_points = np.array(corners, dtype=np.float32)
        size = 400
        dst_points = np.array([
            [0, 0], [size, 0], [size, size], [0, size]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (size, size))
        return warped
    except Exception as e:
        logger.warning(f"Error warping board: {e}")
        return None

def fen_to_board(fen):
    """Convert FEN string to 8x8 board representation"""
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    parts = fen.split()
    if not parts:
        return board
    
    ranks = parts[0].split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                if file_idx < 8:
                    board[rank_idx][file_idx] = char
                    file_idx += 1
    
    return board

def create_test_dataset(test_split=0.3):
    """Create a test dataset from Marshall annotations"""
    logger.info("Creating occupancy test dataset...")
    
    # Load Marshall annotations
    with open("marshall_chess_annotations/annotations.json", 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    
    # Filter out excluded images
    valid_annotations = {
        k: v for k, v in annotations.items() 
        if k not in excluded_images
    }
    
    # Split into train/test (use same random state as training)
    image_names = list(valid_annotations.keys())
    train_names, test_names = train_test_split(
        image_names, test_size=test_split, random_state=42
    )
    
    logger.info(f"Using {len(test_names)} images for testing")
    
    dataset = []
    marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    
    processed = 0
    errors = 0
    
    for image_name in test_names:
        image_path = marshall_photos_dir / image_name
        if not image_path.exists():
            continue
            
        try:
            # Load image
            if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                continue
                
            annotation = valid_annotations[image_name]
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                continue
            
            # Warp board
            warped_board = warp_board(image, corners)
            if warped_board is None:
                continue
            
            # Extract squares and create labels
            board = fen_to_board(fen)
            square_size = warped_board.shape[0] // 8
            
            for rank in range(8):
                for file in range(8):
                    y1 = rank * square_size
                    y2 = (rank + 1) * square_size
                    x1 = file * square_size
                    x2 = (file + 1) * square_size
                    
                    square = warped_board[y1:y2, x1:x2]
                    piece = board[rank][file]
                    
                    # 1 if occupied, 0 if empty
                    label = 1 if piece != '.' else 0
                    
                    # Preprocess square
                    square_tensor = preprocess_square_for_occupancy(square)
                    dataset.append({
                        'square': square_tensor,
                        'label': torch.tensor(label, dtype=torch.long),
                        'image_name': image_name
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} test images...")
    
    logger.info(f"Created occupancy test dataset with {len(dataset)} samples")
    logger.info(f"Processed: {processed}, Errors: {errors}")
    return dataset

def test_marshall_model():
    """Test the Marshall occupancy model on unseen data"""
    logger.info("🔍 Testing Marshall Occupancy Model")
    logger.info("=" * 50)
    
    # Check if Marshall model exists
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if not marshall_path.exists():
        logger.error("Marshall occupancy model not found!")
        return
    
    # Create test dataset
    test_data = create_test_dataset()
    if len(test_data) < 100:
        logger.error("Not enough test data")
        return
    
    logger.info(f"Testing on {len(test_data)} unseen squares")
    
    # Load the Marshall model weights
    try:
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Marshall model: {e}")
        return
    
    # We need to create a simple model architecture to load the weights
    # Let's create a basic ResNet-like architecture
    class SimpleResNet(nn.Module):
        def __init__(self):
            super(SimpleResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # ResNet blocks
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 2)  # 2 classes: occupied/empty
            
        def _make_layer(self, inplanes, planes, blocks, stride=1):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    # Create model and load weights
    model = SimpleResNet()
    
    try:
        model.load_state_dict(marshall_weights)
        model.eval()
        logger.info("✅ Model architecture loaded and weights applied")
    except Exception as e:
        logger.warning(f"Could not load weights into simple architecture: {e}")
        logger.info("This might be due to architecture mismatch, but the model exists")
        return
    
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    correct = 0
    total = 0
    occupied_correct = 0
    occupied_total = 0
    empty_correct = 0
    empty_total = 0
    
    logger.info("Running inference on test data...")
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            square = sample['square'].unsqueeze(0).to(device)
            label = sample['label'].to(device)
            
            output = model(square)
            _, predicted = torch.max(output.data, 1)
            
            total += 1
            correct += (predicted == label).sum().item()
            
            if label.item() == 1:  # Occupied
                occupied_total += 1
                occupied_correct += (predicted == label).sum().item()
            else:  # Empty
                empty_total += 1
                empty_correct += (predicted == label).sum().item()
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(test_data)} samples...")
    
    # Calculate results
    overall_accuracy = 100 * correct / total
    occupied_accuracy = 100 * occupied_correct / occupied_total if occupied_total > 0 else 0
    empty_accuracy = 100 * empty_correct / empty_total if empty_total > 0 else 0
    
    logger.info(f"\n📊 MARSHALL MODEL TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Overall Accuracy:    {overall_accuracy:.2f}% ({correct}/{total})")
    logger.info(f"Occupied Accuracy:   {occupied_accuracy:.2f}% ({occupied_correct}/{occupied_total})")
    logger.info(f"Empty Accuracy:      {empty_accuracy:.2f}% ({empty_correct}/{empty_total})")
    
    # Assessment
    if overall_accuracy > 95:
        logger.info(f"✅ EXCELLENT: Model shows {overall_accuracy:.2f}% accuracy on unseen data!")
        logger.info("   This indicates the Marshall training was successful and not overfitting.")
    elif overall_accuracy > 90:
        logger.info(f"✅ GOOD: Model shows {overall_accuracy:.2f}% accuracy on unseen data.")
        logger.info("   The Marshall training appears to be working well.")
    elif overall_accuracy > 80:
        logger.info(f"⚠️  MODERATE: Model shows {overall_accuracy:.2f}% accuracy on unseen data.")
        logger.info("   The model is learning but may need more training data.")
    else:
        logger.warning(f"❌ POOR: Model shows {overall_accuracy:.2f}% accuracy on unseen data.")
        logger.warning("   This may indicate overfitting or insufficient training.")
    
    return overall_accuracy

def main():
    """Run the simple Marshall model test"""
    logger.info("🧪 Simple Marshall Model Validation")
    logger.info("Testing occupancy model on unseen Marshall data")
    logger.info("=" * 60)
    
    accuracy = test_marshall_model()
    
    if accuracy:
        logger.info(f"\n🎯 Final Assessment:")
        if accuracy > 95:
            logger.info("🎉 Marshall model is performing excellently on real-world data!")
        elif accuracy > 90:
            logger.info("✅ Marshall model is performing well on real-world data!")
        else:
            logger.info("⚠️  Marshall model may need more training or different approach.")

if __name__ == "__main__":
    main()
