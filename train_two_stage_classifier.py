#!/usr/bin/env python3
"""
Training script for the two-stage piece classifier.
This will train both the color classifier and piece type classifier using your existing dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
from pathlib import Path
import logging
import time
from collections import defaultdict
import json

# Import the two-stage classifier
from two_stage_piece_classifier import TwoStagePieceClassifier, ColorClassifier, PieceTypeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessPieceDataset(Dataset):
    """Dataset for training the two-stage piece classifier."""
    
    def __init__(self, data_dir, transform=None, stage="both"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.stage = stage  # "color", "piece_type", or "both"
        
        # Class mappings
        self.color_classes = ['white', 'black']
        self.piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        # Load dataset
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {stage} training")
    
    def _load_samples(self):
        """Load samples from the dataset directory."""
        samples = []
        
        # Expected directory structure: data_dir/color_piece_type/image.png
        for color in self.color_classes:
            for piece_type in self.piece_types:
                class_dir = self.data_dir / f"{color}_{piece_type}"
                if class_dir.exists():
                    images = list(class_dir.glob("*.png"))
                    for img_path in images:
                        samples.append({
                            'image_path': img_path,
                            'color': color,
                            'piece_type': piece_type,
                            'full_class': f"{color}_{piece_type}"
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(str(sample['image_path']))
        if img is None:
            # Return a placeholder if image loading fails
            img = np.zeros((100, 200, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Return appropriate labels based on training stage
        if self.stage == "color":
            color_idx = self.color_classes.index(sample['color'])
            return img, color_idx
        elif self.stage == "piece_type":
            piece_idx = self.piece_types.index(sample['piece_type'])
            return img, piece_idx
        else:  # "both" - return both labels
            color_idx = self.color_classes.index(sample['color'])
            piece_idx = self.piece_types.index(sample['piece_type'])
            return img, (color_idx, piece_idx)

def train_color_classifier(data_dir, model_save_dir="two_stage_models"):
    """Train the color classifier (Stage 1)."""
    print("🎨 Training Color Classifier (Stage 1)")
    print("=" * 50)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ChessPieceDataset(data_dir, transform, stage="color")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorClassifier().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    num_epochs = 20
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch accuracy
        epoch_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}, '
              f'Accuracy: {epoch_accuracy:.2f}%')
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), f"{model_save_dir}/color_classifier.pt")
            print(f"✅ New best color classifier saved! Accuracy: {best_accuracy:.2f}%")
        
        scheduler.step()
    
    print(f"\n🎉 Color classifier training completed! Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy

def train_piece_type_classifier(data_dir, model_save_dir="two_stage_models"):
    """Train the piece type classifier (Stage 2)."""
    print("♟️  Training Piece Type Classifier (Stage 2)")
    print("=" * 50)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ChessPieceDataset(data_dir, transform, stage="piece_type")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PieceTypeClassifier().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    num_epochs = 25
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch accuracy
        epoch_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}, '
              f'Accuracy: {epoch_accuracy:.2f}%')
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), f"{model_save_dir}/piece_type_classifier.pt")
            print(f"✅ New best piece type classifier saved! Accuracy: {best_accuracy:.2f}%")
        
        scheduler.step()
    
    print(f"\n🎉 Piece type classifier training completed! Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy

def main():
    """Main training function."""
    print("🚀 Training Two-Stage Piece Classifier")
    print("=" * 60)
    
    # Configuration
    data_dir = "grey_background_dataset/pieces/train"  # Use training split
    model_save_dir = "two_stage_models"
    
    # Create model directory
    Path(model_save_dir).mkdir(exist_ok=True)
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    print(f"📁 Dataset directory: {data_dir}")
    print(f"💾 Model save directory: {model_save_dir}")
    
    # Training results
    results = {}
    
    try:
        # Stage 1: Train color classifier
        print(f"\n{'='*60}")
        color_accuracy = train_color_classifier(data_dir, model_save_dir)
        results['color_classifier'] = color_accuracy
        
        # Stage 2: Train piece type classifier
        print(f"\n{'='*60}")
        piece_accuracy = train_piece_type_classifier(data_dir, model_save_dir)
        results['piece_type_classifier'] = piece_accuracy
        
        # Calculate expected combined accuracy
        # Assuming independence: P(correct) = P(color_correct) * P(piece_correct)
        expected_combined = (color_accuracy / 100) * (piece_accuracy / 100) * 100
        
        print(f"\n{'='*60}")
        print("📊 TRAINING RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"🎨 Color Classifier Accuracy: {color_accuracy:.2f}%")
        print(f"♟️  Piece Type Classifier Accuracy: {piece_accuracy:.2f}%")
        print(f"🎯 Expected Combined Accuracy: {expected_combined:.2f}%")
        print(f"📈 Improvement over current: +{expected_combined - 13.3:.1f}%")
        
        # Save results
        results['expected_combined_accuracy'] = expected_combined
        results['improvement_over_current'] = expected_combined - 13.3
        results['training_timestamp'] = time.time()
        
        with open(f"{model_save_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Training results saved to: {model_save_dir}/training_results.json")
        
        # Test the trained classifier
        print(f"\n🧪 Testing trained classifier...")
        classifier = TwoStagePieceClassifier(model_save_dir)
        
        # Test with a sample image
        test_dir = Path(data_dir) / "test"
        if test_dir.exists():
            test_images = list(test_dir.glob("*/**/*.png"))
            if test_images:
                test_img_path = test_images[0]
                print(f"Testing with: {test_img_path.name}")
                
                img = cv2.imread(str(test_img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    piece_name, confidence, color_conf, piece_conf = classifier.classify_piece(img)
                    
                    print(f"Result: {piece_name}")
                    print(f"Overall confidence: {confidence:.3f}")
                    print(f"Color confidence: {color_conf:.3f}")
                    print(f"Piece type confidence: {piece_conf:.3f}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Your API now has a two-stage piece classifier with expected accuracy: {expected_combined:.1f}%")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training error: {e}")

if __name__ == "__main__":
    main()
