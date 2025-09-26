#!/usr/bin/env python3
"""
Test Marshall occupancy model on different splits of Marshall data
to check for overfitting and performance differences
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_marshall_occupancy_model():
    """Load the Marshall occupancy model"""
    try:
        # Load original model architecture
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            logger.error(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original model architecture loaded")
        
        # Load Marshall weights
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            logger.error(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        logger.info("✅ Marshall weights loaded")
        
        # Apply Marshall weights
        model.load_state_dict(marshall_weights)
        logger.info("✅ Marshall weights applied to model")
        
        model.eval()
        return model
    except Exception as e:
        logger.error(f"❌ Error loading Marshall model: {e}")
        return None

def load_marshall_data():
    """Load Marshall training data"""
    try:
        annotations_path = Path("marshall_chess_annotations/annotations.json")
        if not annotations_path.exists():
            logger.error(f"❌ Marshall annotations not found at {annotations_path}")
            return None, None
        
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', data)
        logger.info(f"✅ Loaded {len(annotations)} Marshall annotations")
        
        # Create dataset
        images = []
        labels = []
        image_paths = []
        
        for img_path, annotation_data in annotations.items():
            if 'corners' in annotation_data and 'fen' in annotation_data:
                try:
                    # Load image
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 224x224
                    img = img.resize((224, 224))
                    
                    # Convert to tensor
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img)
                    
                    # Create occupancy labels from FEN
                    fen = annotation_data['fen']
                    occupancy_labels = create_occupancy_labels_from_fen(fen)
                    
                    images.append(img_tensor)
                    labels.append(occupancy_labels)
                    image_paths.append(img_path)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {img_path}: {e}")
                    continue
        
        logger.info(f"✅ Created Marshall dataset: {len(images)} images")
        return images, labels, image_paths
        
    except Exception as e:
        logger.error(f"❌ Error loading Marshall data: {e}")
        return None, None, None

def create_occupancy_labels_from_fen(fen):
    """Create occupancy labels from FEN string"""
    try:
        # Parse FEN to get board state
        board_part = fen.split()[0]
        occupancy = []
        
        for char in board_part:
            if char == '/':
                continue
            elif char.isdigit():
                # Empty squares
                occupancy.extend([0] * int(char))
            else:
                # Occupied square
                occupancy.append(1)
        
        # Ensure we have exactly 64 squares
        if len(occupancy) != 64:
            logger.warning(f"⚠️ FEN parsing resulted in {len(occupancy)} squares, expected 64")
            # Pad or truncate to 64
            if len(occupancy) < 64:
                occupancy.extend([0] * (64 - len(occupancy)))
            else:
                occupancy = occupancy[:64]
        
        return occupancy
    except Exception as e:
        logger.warning(f"⚠️ Error parsing FEN {fen}: {e}")
        return [0] * 64

def test_model_on_data(model, images, labels, dataset_name, max_samples=None):
    """Test model accuracy on a dataset"""
    if not images or not labels:
        logger.error(f"❌ No data available for {dataset_name}")
        return None
    
    try:
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Limit samples if specified
        if max_samples and len(images) > max_samples:
            indices = random.sample(range(len(images)), max_samples)
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        with torch.no_grad():
            for i, (img, label) in enumerate(zip(images, labels)):
                if i % 50 == 0:
                    logger.info(f"Testing {dataset_name}: {i}/{len(images)}")
                
                # Convert label to tensor
                label_tensor = torch.tensor(label, dtype=torch.long)
                
                # Get prediction
                img_batch = img.unsqueeze(0)
                outputs = model(img_batch)
                
                # Get predictions for all 64 squares
                if outputs.dim() > 1 and outputs.shape[1] == 2:
                    # Binary classification for each square
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    # Single output per square
                    predictions = (outputs > 0.5).long().squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label_tensor.numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        
        logger.info(f"✅ {dataset_name} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"❌ Error testing {dataset_name}: {e}")
        return None

def create_data_splits(images, labels, image_paths):
    """Create different splits of the Marshall data"""
    try:
        # Split 1: Random 70/30 split
        train_imgs, test_imgs, train_labels, test_labels, train_paths, test_paths = train_test_split(
            images, labels, image_paths, test_size=0.3, random_state=42
        )
        
        # Split 2: First half vs second half (temporal)
        mid_point = len(images) // 2
        first_half_imgs = images[:mid_point]
        first_half_labels = labels[:mid_point]
        second_half_imgs = images[mid_point:]
        second_half_labels = labels[mid_point:]
        
        # Split 3: Random sample (20% of data)
        sample_size = max(10, len(images) // 5)
        sample_indices = random.sample(range(len(images)), sample_size)
        sample_imgs = [images[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        
        return {
            'random_train': (train_imgs, train_labels),
            'random_test': (test_imgs, test_labels),
            'first_half': (first_half_imgs, first_half_labels),
            'second_half': (second_half_imgs, second_half_labels),
            'random_sample': (sample_imgs, sample_labels)
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating data splits: {e}")
        return None

def main():
    """Main comparison function"""
    logger.info("🔍 Testing Marshall Model on Different Data Splits")
    
    # Load model
    logger.info("📥 Loading Marshall model...")
    model = load_marshall_occupancy_model()
    
    if not model:
        logger.error("❌ Failed to load Marshall model")
        return
    
    # Load data
    logger.info("📥 Loading Marshall data...")
    images, labels, image_paths = load_marshall_data()
    
    if not images or not labels:
        logger.error("❌ Failed to load Marshall data")
        return
    
    # Create data splits
    logger.info("📊 Creating data splits...")
    splits = create_data_splits(images, labels, image_paths)
    
    if not splits:
        logger.error("❌ Failed to create data splits")
        return
    
    # Test model on different splits
    logger.info("🧪 Testing model on different data splits...")
    
    results = {}
    
    # Test on random train/test split
    logger.info("\n--- Random Train/Test Split ---")
    train_acc = test_model_on_data(model, *splits['random_train'], "Random Train Set")
    test_acc = test_model_on_data(model, *splits['random_test'], "Random Test Set")
    
    if train_acc is not None and test_acc is not None:
        results['random_train'] = train_acc
        results['random_test'] = test_acc
        overfitting_gap = train_acc - test_acc
        logger.info(f"📈 Overfitting Gap (Train - Test): {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
    
    # Test on temporal splits
    logger.info("\n--- Temporal Split (First Half vs Second Half) ---")
    first_half_acc = test_model_on_data(model, *splits['first_half'], "First Half (Earlier Data)")
    second_half_acc = test_model_on_data(model, *splits['second_half'], "Second Half (Later Data)")
    
    if first_half_acc is not None and second_half_acc is not None:
        results['first_half'] = first_half_acc
        results['second_half'] = second_half_acc
        temporal_difference = abs(first_half_acc - second_half_acc)
        logger.info(f"📈 Temporal Consistency: {temporal_difference:.4f} ({temporal_difference*100:.2f}%) difference")
    
    # Test on random sample
    logger.info("\n--- Random Sample Test ---")
    sample_acc = test_model_on_data(model, *splits['random_sample'], "Random Sample")
    
    if sample_acc is not None:
        results['random_sample'] = sample_acc
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 MARSHALL DATA SPLIT TEST RESULTS")
    logger.info("="*60)
    
    for split_name, accuracy in results.items():
        logger.info(f"🎯 {split_name.replace('_', ' ').title()}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate statistics
    if len(results) > 1:
        accuracies = list(results.values())
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        logger.info(f"\n📈 Statistics:")
        logger.info(f"   Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
        logger.info(f"   Std Deviation: {std_acc:.4f} ({std_acc*100:.2f}%)")
        logger.info(f"   Min Accuracy:  {min_acc:.4f} ({min_acc*100:.2f}%)")
        logger.info(f"   Max Accuracy:  {max_acc:.4f} ({max_acc*100:.2f}%)")
        
        # Overfitting assessment
        if 'random_train' in results and 'random_test' in results:
            overfitting_gap = results['random_train'] - results['random_test']
            if overfitting_gap > 0.05:  # 5% gap
                logger.info(f"⚠️  Potential overfitting detected (gap: {overfitting_gap*100:.2f}%)")
            else:
                logger.info(f"✅ No significant overfitting (gap: {overfitting_gap*100:.2f}%)")
        
        # Consistency assessment
        if std_acc < 0.01:  # 1% standard deviation
            logger.info(f"✅ High consistency across data splits (std: {std_acc*100:.2f}%)")
        else:
            logger.info(f"⚠️  Moderate variation across data splits (std: {std_acc*100:.2f}%)")
    
    logger.info("\n✅ Marshall data split testing complete!")

if __name__ == "__main__":
    main()
