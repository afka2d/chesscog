#!/usr/bin/env python3
"""
Compare Marshall occupancy model accuracy on Marshall data vs original training data
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
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_original_occupancy_model():
    """Load the original occupancy model for comparison"""
    try:
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            logger.error(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original model loaded")
        model.eval()
        return model
    except Exception as e:
        logger.error(f"❌ Error loading original model: {e}")
        return None

def load_marshall_data():
    """Load Marshall training data"""
    try:
        annotations_path = Path("marshall_chess_annotations/annotations.json")
        if not annotations_path.exists():
            logger.error(f"❌ Marshall annotations not found at {annotations_path}")
            return None, None
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        logger.info(f"✅ Loaded {len(annotations)} Marshall annotations")
        
        # Create dataset
        images = []
        labels = []
        
        for img_path, data in annotations.items():
            if 'corners' in data and 'fen' in data:
                try:
                    # Load image
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 224x224
                    img = img.resize((224, 224))
                    img_array = np.array(img)
                    
                    # Convert to tensor
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img)
                    
                    # Extract corners
                    corners = np.array(data['corners'], dtype=np.float32)
                    
                    # Create occupancy labels from FEN
                    fen = data['fen']
                    occupancy_labels = create_occupancy_labels_from_fen(fen)
                    
                    images.append(img_tensor)
                    labels.append(occupancy_labels)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {img_path}: {e}")
                    continue
        
        logger.info(f"✅ Created Marshall dataset: {len(images)} images")
        return images, labels
        
    except Exception as e:
        logger.error(f"❌ Error loading Marshall data: {e}")
        return None, None

def load_original_data():
    """Load original training data"""
    try:
        # Look for original training data
        data_paths = [
            "data/train/occupancy",
            "data/occupancy/train",
            "runs/occupancy_classifier/data/train"
        ]
        
        images = []
        labels = []
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                logger.info(f"✅ Found original data at {path}")
                
                # Load images and labels
                for img_file in path.glob("*.jpg"):
                    try:
                        img = Image.open(img_file)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img = img.resize((224, 224))
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = transform(img)
                        
                        # Extract label from filename or directory structure
                        label = extract_label_from_path(img_file)
                        if label is not None:
                            images.append(img_tensor)
                            labels.append(label)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {img_file}: {e}")
                        continue
                
                break
        
        if not images:
            logger.warning("⚠️ No original training data found")
            return None, None
        
        logger.info(f"✅ Created original dataset: {len(images)} images")
        return images, labels
        
    except Exception as e:
        logger.error(f"❌ Error loading original data: {e}")
        return None, None

def extract_label_from_path(img_path):
    """Extract occupancy label from image path"""
    try:
        # Try to extract from directory structure
        parts = str(img_path).split('/')
        for part in parts:
            if 'occupied' in part.lower():
                return 1
            elif 'empty' in part.lower():
                return 0
        
        # Try to extract from filename
        filename = img_path.stem.lower()
        if 'occupied' in filename:
            return 1
        elif 'empty' in filename:
            return 0
        
        return None
    except:
        return None

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

def test_model_accuracy(model, images, labels, dataset_name):
    """Test model accuracy on a dataset"""
    if not images or not labels:
        logger.error(f"❌ No data available for {dataset_name}")
        return None, None
    
    try:
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i, (img, label) in enumerate(zip(images, labels)):
                if i % 100 == 0:
                    logger.info(f"Testing {dataset_name}: {i}/{len(images)}")
                
                # Handle different label formats
                if isinstance(label, list):
                    # Marshall data - 64 squares
                    label_tensor = torch.tensor(label, dtype=torch.long)
                else:
                    # Original data - single label
                    label_tensor = torch.tensor([label], dtype=torch.long)
                
                # Get prediction
                img_batch = img.unsqueeze(0)
                outputs = model(img_batch)
                
                if outputs.dim() > 1:
                    # Multi-class output
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    # Binary output
                    predictions = (outputs > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label_tensor.numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        
        logger.info(f"✅ {dataset_name} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy, (all_labels, all_predictions)
        
    except Exception as e:
        logger.error(f"❌ Error testing {dataset_name}: {e}")
        return None, None

def create_comparison_plot(marshall_results, original_results):
    """Create comparison plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        datasets = ['Marshall Data', 'Original Data']
        accuracies = [marshall_results[0], original_results[0]]
        
        bars = ax1.bar(datasets, accuracies, color=['#2E8B57', '#4169E1'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Marshall Model Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}\n({acc*100:.2f}%)', 
                    ha='center', va='bottom')
        
        # Confusion matrices
        if marshall_results[1] and original_results[1]:
            from sklearn.metrics import confusion_matrix
            
            # Marshall confusion matrix
            marshall_labels, marshall_preds = marshall_results[1]
            marshall_cm = confusion_matrix(marshall_labels, marshall_preds)
            sns.heatmap(marshall_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Marshall Data Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('marshall_vs_original_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("✅ Comparison plot saved as 'marshall_vs_original_accuracy_comparison.png'")
        
    except Exception as e:
        logger.error(f"❌ Error creating comparison plot: {e}")

def main():
    """Main comparison function"""
    logger.info("🔍 Starting Marshall vs Original Data Accuracy Comparison")
    
    # Load models
    logger.info("📥 Loading models...")
    marshall_model = load_marshall_occupancy_model()
    original_model = load_original_occupancy_model()
    
    if not marshall_model:
        logger.error("❌ Failed to load Marshall model")
        return
    
    if not original_model:
        logger.error("❌ Failed to load original model")
        return
    
    # Load datasets
    logger.info("📥 Loading datasets...")
    marshall_images, marshall_labels = load_marshall_data()
    original_images, original_labels = load_original_data()
    
    if not marshall_images or not marshall_labels:
        logger.error("❌ Failed to load Marshall data")
        return
    
    if not original_images or not original_labels:
        logger.error("❌ Failed to load original data")
        return
    
    # Test Marshall model on both datasets
    logger.info("🧪 Testing Marshall model on Marshall data...")
    marshall_on_marshall = test_model_accuracy(marshall_model, marshall_images, marshall_labels, "Marshall Model on Marshall Data")
    
    logger.info("🧪 Testing Marshall model on original data...")
    marshall_on_original = test_model_accuracy(marshall_model, original_images, original_labels, "Marshall Model on Original Data")
    
    # Test original model on both datasets for comparison
    logger.info("🧪 Testing original model on Marshall data...")
    original_on_marshall = test_model_accuracy(original_model, marshall_images, marshall_labels, "Original Model on Marshall Data")
    
    logger.info("🧪 Testing original model on original data...")
    original_on_original = test_model_accuracy(original_model, original_images, original_labels, "Original Model on Original Data")
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("📊 ACCURACY COMPARISON RESULTS")
    logger.info("="*60)
    
    if marshall_on_marshall[0] is not None:
        logger.info(f"🎯 Marshall Model on Marshall Data: {marshall_on_marshall[0]:.4f} ({marshall_on_marshall[0]*100:.2f}%)")
    
    if marshall_on_original[0] is not None:
        logger.info(f"🎯 Marshall Model on Original Data: {marshall_on_original[0]:.4f} ({marshall_on_original[0]*100:.2f}%)")
    
    if original_on_marshall[0] is not None:
        logger.info(f"🎯 Original Model on Marshall Data: {original_on_marshall[0]:.4f} ({original_on_marshall[0]*100:.2f}%)")
    
    if original_on_original[0] is not None:
        logger.info(f"🎯 Original Model on Original Data: {original_on_original[0]:.4f} ({original_on_original[0]*100:.2f}%)")
    
    # Calculate differences
    if marshall_on_marshall[0] is not None and marshall_on_original[0] is not None:
        difference = marshall_on_marshall[0] - marshall_on_original[0]
        logger.info(f"\n📈 Marshall Model Performance Difference:")
        logger.info(f"   Marshall Data vs Original Data: {difference:+.4f} ({difference*100:+.2f}%)")
        
        if difference > 0:
            logger.info("   ✅ Marshall model performs BETTER on Marshall data")
        elif difference < 0:
            logger.info("   ⚠️ Marshall model performs WORSE on Marshall data")
        else:
            logger.info("   ➖ Marshall model performs EQUALLY on both datasets")
    
    # Create comparison plot
    if marshall_on_marshall[0] is not None and marshall_on_original[0] is not None:
        create_comparison_plot(marshall_on_marshall, marshall_on_original)
    
    logger.info("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()
