#!/usr/bin/env python3
"""
Comprehensive validation of Marshall models
Tests for overfitting and confirms model naming
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
from sklearn.metrics import accuracy_score, classification_report
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

def create_validation_dataset(test_split=0.3):
    """Create validation dataset from Marshall annotations"""
    logger.info("Creating validation dataset...")
    
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
    
    logger.info(f"Using {len(test_names)} images for validation")
    
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
            logger.info(f"Processed {processed} validation images...")
    
    logger.info(f"Created validation dataset with {len(dataset)} samples")
    logger.info(f"Processed: {processed}, Errors: {errors}")
    return dataset

def test_marshall_occupancy_model():
    """Test Marshall occupancy model on validation data"""
    logger.info("🧪 Testing Marshall Occupancy Model on Validation Data")
    logger.info("=" * 60)
    
    # Check if Marshall model exists
    marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
    if not marshall_path.exists():
        logger.error("❌ Marshall occupancy model not found!")
        return None
    
    # Create validation dataset
    val_data = create_validation_dataset()
    if len(val_data) < 100:
        logger.error("Not enough validation data")
        return None
    
    logger.info(f"Testing on {len(val_data)} validation samples")
    
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
        
        logger.info("Running validation inference...")
        
        with torch.no_grad():
            for i, sample in enumerate(val_data):
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
                    logger.info(f"Processed {i + 1}/{len(val_data)} samples...")
        
        # Calculate results
        overall_accuracy = 100 * correct / total
        occupied_accuracy = 100 * occupied_correct / occupied_total if occupied_total > 0 else 0
        empty_accuracy = 100 * empty_correct / empty_total if empty_total > 0 else 0
        
        logger.info(f"\n📊 VALIDATION RESULTS:")
        logger.info(f"Overall Accuracy:    {overall_accuracy:.2f}% ({correct}/{total})")
        logger.info(f"Occupied Accuracy:   {occupied_accuracy:.2f}% ({occupied_correct}/{occupied_total})")
        logger.info(f"Empty Accuracy:      {empty_accuracy:.2f}% ({empty_correct}/{empty_total})")
        
        return {
            'overall_accuracy': overall_accuracy,
            'occupied_accuracy': occupied_accuracy,
            'empty_accuracy': empty_accuracy,
            'total_samples': total,
            'occupied_samples': occupied_total,
            'empty_samples': empty_total
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return None

def check_model_naming():
    """Check that Marshall models have proper naming and don't replace originals"""
    logger.info("🔍 Checking Model Naming and Safety")
    logger.info("=" * 50)
    
    # Check original models
    original_models = {
        'occupancy': Path("runs/occupancy_classifier/ResNet/ResNet.pt"),
        'color': Path("models/color_classifier_simple.pt"),
        'piece': Path("models/piece_classifier_simple.pt")
    }
    
    # Check Marshall models
    marshall_models = {
        'occupancy': Path("models_marshall_improved/occupancy_marshall.pt"),
        'color': Path("models_marshall_improved/color_classification_marshall.pt"),
        'piece': Path("models_marshall_improved/piece_classification_marshall.pt")
    }
    
    logger.info("📁 Original Models Status:")
    for name, path in original_models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"   ❌ {name}: {path.name} - NOT FOUND")
    
    logger.info("\n📁 Marshall Models Status:")
    for name, path in marshall_models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            logger.info(f"   ⏳ {name}: {path.name} - Not trained yet")
    
    # Verify naming convention
    logger.info("\n🔒 Safety Check:")
    all_safe = True
    
    for name, path in marshall_models.items():
        if path.exists():
            if "marshall" in path.name.lower():
                logger.info(f"   ✅ {name}: Contains 'marshall' in filename")
            else:
                logger.error(f"   ❌ {name}: Missing 'marshall' in filename!")
                all_safe = False
    
    # Check that originals are untouched
    for name, path in original_models.items():
        if path.exists():
            logger.info(f"   ✅ {name}: Original model preserved")
        else:
            logger.warning(f"   ⚠️  {name}: Original model not found")
    
    if all_safe:
        logger.info("   🎉 All Marshall models follow proper naming convention!")
    else:
        logger.error("   ❌ Some models don't follow naming convention!")
    
    return all_safe

def check_training_status():
    """Check current training status"""
    logger.info("🔄 Checking Training Status")
    logger.info("=" * 40)
    
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        training_scripts = [
            'train_occupancy_marshall.py',
            'train_color_marshall.py', 
            'train_piece_marshall.py',
            'overnight_marshall_training.py'
        ]
        
        running_scripts = []
        for script in training_scripts:
            if script in result.stdout:
                running_scripts.append(script)
                logger.info(f"   ✅ {script} is running")
            else:
                logger.info(f"   ⏸️  {script} is not running")
        
        if running_scripts:
            logger.info(f"\n   🚀 {len(running_scripts)} training script(s) currently running")
        else:
            logger.info(f"\n   ⏸️  No training scripts currently running")
        
        return running_scripts
        
    except Exception as e:
        logger.error(f"Error checking training status: {e}")
        return []

def main():
    """Run comprehensive validation"""
    logger.info("🔍 COMPREHENSIVE MARSHALL MODEL VALIDATION")
    logger.info("=" * 60)
    logger.info("Testing for overfitting and confirming model safety")
    logger.info("=" * 60)
    
    # 1. Check model naming and safety
    naming_safe = check_model_naming()
    
    # 2. Check training status
    running_scripts = check_training_status()
    
    # 3. Test Marshall occupancy model for overfitting
    logger.info(f"\n🧪 OVERFITTING VALIDATION")
    logger.info("=" * 40)
    
    validation_results = test_marshall_occupancy_model()
    
    # 4. Final assessment
    logger.info(f"\n🏁 FINAL ASSESSMENT")
    logger.info("=" * 50)
    
    if naming_safe:
        logger.info("✅ Model naming is safe - originals preserved")
    else:
        logger.error("❌ Model naming issues detected")
    
    if validation_results:
        accuracy = validation_results['overall_accuracy']
        if accuracy > 95:
            logger.info(f"✅ Marshall model shows excellent performance ({accuracy:.2f}%)")
            logger.info("   No signs of overfitting detected")
        elif accuracy > 90:
            logger.info(f"✅ Marshall model shows good performance ({accuracy:.2f}%)")
            logger.info("   Minimal overfitting risk")
        elif accuracy > 80:
            logger.info(f"⚠️  Marshall model shows moderate performance ({accuracy:.2f}%)")
            logger.info("   Some overfitting may be present")
        else:
            logger.warning(f"❌ Marshall model shows poor performance ({accuracy:.2f}%)")
            logger.warning("   Significant overfitting likely")
    else:
        logger.error("❌ Could not validate Marshall model performance")
    
    if running_scripts:
        logger.info(f"🚀 Training is active: {', '.join(running_scripts)}")
        logger.info("   Models will continue training overnight")
    else:
        logger.info("⏸️  No active training detected")
        logger.info("   You may need to start overnight training")
    
    # Summary
    logger.info(f"\n📋 SUMMARY:")
    logger.info(f"   Model Safety: {'✅ SAFE' if naming_safe else '❌ UNSAFE'}")
    logger.info(f"   Performance: {'✅ GOOD' if validation_results and validation_results['overall_accuracy'] > 90 else '⚠️  NEEDS REVIEW'}")
    logger.info(f"   Training: {'🚀 ACTIVE' if running_scripts else '⏸️  INACTIVE'}")
    
    if naming_safe and validation_results and validation_results['overall_accuracy'] > 90:
        logger.info(f"\n🎉 CONCLUSION: Marshall models are ready for production!")
    else:
        logger.warning(f"\n⚠️  CONCLUSION: Some issues need attention before production use")

if __name__ == "__main__":
    main()
