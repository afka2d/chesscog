#!/usr/bin/env python3
"""
Comprehensive overfitting test for Marshall occupancy model
Tests on different data splits to detect overfitting
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
from sklearn.model_selection import train_test_split
import random

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

def create_dataset_split(image_names, test_split=0.3, random_state=42):
    """Create train/test split with specific random state"""
    return train_test_split(image_names, test_size=test_split, random_state=random_state)

def create_dataset_from_images(image_names, dataset_name):
    """Create dataset from specific image names"""
    logger.info(f"Creating {dataset_name} dataset from {len(image_names)} images...")
    
    # Load Marshall annotations
    with open("marshall_chess_annotations/annotations.json", 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    
    dataset = []
    marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    
    processed = 0
    errors = 0
    
    for image_name in image_names:
        if image_name in excluded_images:
            continue
            
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
                
            annotation = annotations.get(image_name, {})
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
                        'label': label,
                        'image_name': image_name,
                        'position': f"{chr(ord('a') + file)}{8 - rank}"
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} {dataset_name} images...")
    
    logger.info(f"Created {dataset_name} dataset with {len(dataset)} squares")
    logger.info(f"Processed: {processed}, Errors: {errors}")
    return dataset

def load_marshall_model():
    """Load Marshall model by combining original architecture with Marshall weights"""
    logger.info("Loading Marshall occupancy model...")
    
    # Load original model architecture
    original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    if not original_model_path.exists():
        logger.error(f"Original model not found: {original_model_path}")
        return None
    
    try:
        # Load original model
        original_model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original model architecture loaded")
        
        # Load Marshall weights
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall weights loaded")
        
        # Apply Marshall weights to original model
        original_model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to model")
        
        return original_model
        
    except Exception as e:
        logger.error(f"Error loading Marshall model: {e}")
        return None

def test_model_on_dataset(model, dataset, dataset_name):
    """Test model on a specific dataset"""
    logger.info(f"Testing on {dataset_name} ({len(dataset)} squares)...")
    
    model.eval()
    
    correct = 0
    total = 0
    occupied_correct = 0
    occupied_total = 0
    empty_correct = 0
    empty_total = 0
    
    errors = []
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            square = sample['square'].unsqueeze(0)
            label = sample['label']
            
            output = model(square)
            _, predicted = torch.max(output.data, 1)
            predicted_label = predicted.item()
            
            total += 1
            is_correct = (predicted_label == label)
            correct += is_correct
            
            if label == 1:  # Occupied
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
                    'true_label': 'occupied' if label == 1 else 'empty',
                    'predicted': 'occupied' if predicted_label == 1 else 'empty'
                })
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} squares...")
    
    # Calculate results
    overall_accuracy = 100 * correct / total
    occupied_accuracy = 100 * occupied_correct / occupied_total if occupied_total > 0 else 0
    empty_accuracy = 100 * empty_correct / empty_total if empty_total > 0 else 0
    
    logger.info(f"📊 {dataset_name} Results:")
    logger.info(f"   Overall Accuracy:    {overall_accuracy:.2f}% ({correct}/{total})")
    logger.info(f"   Occupied Accuracy:   {occupied_accuracy:.2f}% ({occupied_correct}/{occupied_total})")
    logger.info(f"   Empty Accuracy:      {empty_accuracy:.2f}% ({empty_correct}/{empty_total})")
    logger.info(f"   Total Errors:        {len(errors)}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'occupied_accuracy': occupied_accuracy,
        'empty_accuracy': empty_accuracy,
        'total_samples': total,
        'occupied_samples': occupied_total,
        'empty_samples': empty_total,
        'errors': len(errors),
        'error_details': errors[:10]  # First 10 errors for analysis
    }

def comprehensive_overfitting_test():
    """Run comprehensive overfitting test"""
    logger.info("🔍 COMPREHENSIVE OVERFITTING TEST")
    logger.info("=" * 70)
    logger.info("Testing Marshall model on different data splits to detect overfitting")
    logger.info("=" * 70)
    
    # Load Marshall model
    model = load_marshall_model()
    if model is None:
        logger.error("❌ Could not load Marshall model!")
        return None
    
    # Load all available images
    with open("marshall_chess_annotations/annotations.json", 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    
    # Get all valid image names
    all_image_names = [name for name in annotations.keys() if name not in excluded_images]
    logger.info(f"Total available images: {len(all_image_names)}")
    
    # Test 1: Original test split (same as before)
    logger.info(f"\n🧪 TEST 1: Original Test Split (Random State 42)")
    logger.info("-" * 50)
    train_names_42, test_names_42 = create_dataset_split(all_image_names, test_split=0.3, random_state=42)
    test_dataset_42 = create_dataset_from_images(test_names_42, "Original Test")
    results_42 = test_model_on_dataset(model, test_dataset_42, "Original Test")
    
    # Test 2: Different random state (completely different split)
    logger.info(f"\n🧪 TEST 2: Different Random Split (Random State 123)")
    logger.info("-" * 50)
    train_names_123, test_names_123 = create_dataset_split(all_image_names, test_split=0.3, random_state=123)
    test_dataset_123 = create_dataset_from_images(test_names_123, "Different Test")
    results_123 = test_model_on_dataset(model, test_dataset_123, "Different Test")
    
    # Test 3: Different test split size (50% test)
    logger.info(f"\n🧪 TEST 3: Larger Test Split (50% test, Random State 42)")
    logger.info("-" * 50)
    train_names_50, test_names_50 = create_dataset_split(all_image_names, test_split=0.5, random_state=42)
    test_dataset_50 = create_dataset_from_images(test_names_50, "Larger Test")
    results_50 = test_model_on_dataset(model, test_dataset_50, "Larger Test")
    
    # Test 4: Random sample (random 20% of images)
    logger.info(f"\n🧪 TEST 4: Random Sample (20% random images)")
    logger.info("-" * 50)
    random.seed(42)
    random_sample = random.sample(all_image_names, int(len(all_image_names) * 0.2))
    test_dataset_random = create_dataset_from_images(random_sample, "Random Sample")
    results_random = test_model_on_dataset(model, test_dataset_random, "Random Sample")
    
    # Test 5: First half vs second half (temporal split)
    logger.info(f"\n🧪 TEST 5: Temporal Split (Second Half of Images)")
    logger.info("-" * 50)
    sorted_images = sorted(all_image_names)
    second_half = sorted_images[len(sorted_images)//2:]
    test_dataset_temporal = create_dataset_from_images(second_half, "Temporal Test")
    results_temporal = test_model_on_dataset(model, test_dataset_temporal, "Temporal Test")
    
    # Analysis
    logger.info(f"\n📊 OVERFITTING ANALYSIS")
    logger.info("=" * 70)
    
    all_results = {
        "Original Test (30%, R42)": results_42,
        "Different Split (30%, R123)": results_123,
        "Larger Test (50%, R42)": results_50,
        "Random Sample (20%)": results_random,
        "Temporal Split (50%)": results_temporal
    }
    
    accuracies = []
    for test_name, results in all_results.items():
        if results:
            acc = results['overall_accuracy']
            accuracies.append(acc)
            logger.info(f"{test_name:25}: {acc:6.2f}% ({results['total_samples']:4d} samples)")
        else:
            logger.error(f"{test_name:25}: FAILED")
    
    if accuracies:
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = np.std(accuracies)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        
        logger.info(f"\n📈 STATISTICAL ANALYSIS:")
        logger.info(f"   Mean Accuracy:      {mean_accuracy:.2f}%")
        logger.info(f"   Std Deviation:      {std_accuracy:.2f}%")
        logger.info(f"   Min Accuracy:       {min_accuracy:.2f}%")
        logger.info(f"   Max Accuracy:       {max_accuracy:.2f}%")
        logger.info(f"   Accuracy Range:     {max_accuracy - min_accuracy:.2f}%")
        
        # Overfitting assessment
        logger.info(f"\n🎯 OVERFITTING ASSESSMENT:")
        if std_accuracy < 1.0:
            logger.info("   ✅ LOW VARIANCE: Model is consistent across different data splits")
            logger.info("   ✅ NO OVERFITTING: Model generalizes well to different data")
        elif std_accuracy < 3.0:
            logger.info("   ⚠️  MODERATE VARIANCE: Some variation in performance")
            logger.info("   ⚠️  MINOR OVERFITTING: Model may be slightly overfitted")
        else:
            logger.warning("   ❌ HIGH VARIANCE: Large variation in performance")
            logger.warning("   ❌ SIGNIFICANT OVERFITTING: Model is likely overfitted")
        
        if min_accuracy > 95:
            logger.info("   ✅ EXCELLENT: Even worst case performance is very good")
        elif min_accuracy > 90:
            logger.info("   ✅ GOOD: Worst case performance is acceptable")
        else:
            logger.warning("   ⚠️  CONCERNING: Some test splits show poor performance")
    
    return all_results

def main():
    """Run comprehensive overfitting test"""
    logger.info("🔍 MARSHALL MODEL OVERFITTING VALIDATION")
    logger.info("=" * 70)
    logger.info("Testing model on multiple data splits to detect overfitting")
    logger.info("This will help confirm if the 99.70% accuracy is real or overfitting")
    logger.info("=" * 70)
    
    results = comprehensive_overfitting_test()
    
    if results:
        logger.info(f"\n🎉 FINAL OVERFITTING CONCLUSION:")
        logger.info("   The Marshall model has been tested on multiple data splits.")
        logger.info("   This comprehensive test helps verify the accuracy is genuine.")
    else:
        logger.error(f"❌ Overfitting test failed - check for errors above.")

if __name__ == "__main__":
    main()
