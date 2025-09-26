#!/usr/bin/env python3
"""
Test Marshall models on real-world data to show actual accuracy
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
import time

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
    logger.info("Creating real-world test dataset...")
    
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
    
    logger.info(f"Using {len(test_names)} real-world images for testing")
    
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
                        'image_name': image_name,
                        'position': f"{chr(ord('a') + file)}{8 - rank}"
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} test images...")
    
    logger.info(f"Created real-world test dataset with {len(dataset)} squares")
    logger.info(f"Processed: {processed} images, Errors: {errors}")
    return dataset

def test_marshall_occupancy_real_world():
    """Test Marshall occupancy model on real-world data"""
    logger.info("🧪 Testing Marshall Occupancy Model on Real-World Data")
    logger.info("=" * 70)
    
    # Check if Marshall model exists
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if not marshall_path.exists():
        logger.error("❌ Marshall occupancy model not found!")
        return None
    
    # Create test dataset
    test_data = create_test_dataset()
    if len(test_data) < 100:
        logger.error("Not enough test data")
        return None
    
    logger.info(f"Testing on {len(test_data)} real-world chess squares")
    
    # Load the Marshall model
    try:
        model = torch.load(str(marshall_path), map_location='cpu', weights_only=False)
        logger.info("✅ Marshall model loaded successfully")
        
        # Test the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        occupied_correct = 0
        occupied_total = 0
        empty_correct = 0
        empty_total = 0
        
        # Track errors for analysis
        errors = []
        
        logger.info("Running real-world inference...")
        start_time = time.time()
        
        with torch.no_grad():
            for i, sample in enumerate(test_data):
                square = sample['square'].unsqueeze(0).to(device)
                label = sample['label'].to(device)
                
                output = model(square)
                _, predicted = torch.max(output.data, 1)
                
                total += 1
                is_correct = (predicted == label).sum().item()
                correct += is_correct
                
                if label.item() == 1:  # Occupied
                    occupied_total += 1
                    occupied_correct += is_correct
                else:  # Empty
                    empty_total += 1
                    empty_correct += is_correct
                
                # Track errors
                if not is_correct:
                    errors.append({
                        'position': sample['position'],
                        'image': sample['image_name'],
                        'true_label': 'occupied' if label.item() == 1 else 'empty',
                        'predicted': 'occupied' if predicted.item() == 1 else 'empty'
                    })
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} squares...")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate results
        overall_accuracy = 100 * correct / total
        occupied_accuracy = 100 * occupied_correct / occupied_total if occupied_total > 0 else 0
        empty_accuracy = 100 * empty_correct / empty_total if empty_total > 0 else 0
        
        logger.info(f"\n📊 REAL-WORLD ACCURACY RESULTS:")
        logger.info(f"{'='*70}")
        logger.info(f"Overall Accuracy:    {overall_accuracy:.2f}% ({correct}/{total})")
        logger.info(f"Occupied Accuracy:   {occupied_accuracy:.2f}% ({occupied_correct}/{occupied_total})")
        logger.info(f"Empty Accuracy:      {empty_accuracy:.2f}% ({empty_correct}/{empty_total})")
        logger.info(f"Processing Speed:    {len(test_data)/duration:.1f} squares/second")
        logger.info(f"Total Time:          {duration:.1f} seconds")
        
        # Show some error examples
        if errors:
            logger.info(f"\n🔍 ERROR ANALYSIS (showing first 5 errors):")
            for i, error in enumerate(errors[:5]):
                logger.info(f"   {i+1}. Position {error['position']} in {error['image']}: "
                          f"True={error['true_label']}, Predicted={error['predicted']}")
        
        # Assessment
        logger.info(f"\n🎯 REAL-WORLD PERFORMANCE ASSESSMENT:")
        if overall_accuracy > 98:
            logger.info(f"   🏆 EXCELLENT: {overall_accuracy:.2f}% accuracy on real-world data!")
            logger.info("   This is production-ready performance.")
        elif overall_accuracy > 95:
            logger.info(f"   ✅ VERY GOOD: {overall_accuracy:.2f}% accuracy on real-world data!")
            logger.info("   This is excellent performance for chess applications.")
        elif overall_accuracy > 90:
            logger.info(f"   ✅ GOOD: {overall_accuracy:.2f}% accuracy on real-world data!")
            logger.info("   This is solid performance with room for improvement.")
        elif overall_accuracy > 80:
            logger.info(f"   ⚠️  MODERATE: {overall_accuracy:.2f}% accuracy on real-world data!")
            logger.info("   This may need more training data or different approach.")
        else:
            logger.warning(f"   ❌ POOR: {overall_accuracy:.2f}% accuracy on real-world data!")
            logger.warning("   This indicates significant issues.")
        
        return {
            'overall_accuracy': overall_accuracy,
            'occupied_accuracy': occupied_accuracy,
            'empty_accuracy': empty_accuracy,
            'total_samples': total,
            'occupied_samples': occupied_total,
            'empty_samples': empty_total,
            'errors': len(errors),
            'processing_speed': len(test_data)/duration
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return None

def main():
    """Run real-world accuracy test"""
    logger.info("🔍 MARSHALL MODEL REAL-WORLD ACCURACY TEST")
    logger.info("=" * 70)
    logger.info("Testing Marshall occupancy model on unseen real-world data")
    logger.info("This shows actual performance, not just training accuracy")
    logger.info("=" * 70)
    
    results = test_marshall_occupancy_real_world()
    
    if results:
        logger.info(f"\n🎉 FINAL CONCLUSION:")
        logger.info(f"   The Marshall occupancy model achieved {results['overall_accuracy']:.2f}% accuracy")
        logger.info(f"   on {results['total_samples']} real-world chess squares!")
        logger.info(f"   This demonstrates the model works well in practice.")
        
        if results['overall_accuracy'] > 95:
            logger.info(f"   🚀 The model is ready for production use!")
        else:
            logger.info(f"   ⚠️  Consider additional training or data augmentation.")
    else:
        logger.error(f"❌ Could not test the model - check for errors above.")

if __name__ == "__main__":
    main()
