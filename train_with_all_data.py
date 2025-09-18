#!/usr/bin/env python3
"""
Train corner detection model with ALL available training data (231+ annotations).
Focus on simplicity and robustness rather than complexity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import json
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCornerModel(nn.Module):
    """Improved corner detection model with ResNet34 backbone"""
    def __init__(self, backbone='resnet34', pretrained=True):
        super(ImprovedCornerModel, self).__init__()
        
        if backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Improved corner head - more capacity but not too complex
        self.corner_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8)  # 4 corners * (x, y)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

class RobustCornerDataset(Dataset):
    """Robust dataset that loads ALL available data with careful preprocessing"""
    def __init__(self, data, image_size=384, augment=True):
        self.data = data
        self.image_size = image_size
        self.augment = augment
        
        # Robust transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Conservative augmentation (only when we're sure it won't break corners)
        self.augment_transforms = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image with error handling
        try:
            image = cv2.imread(item['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {item['image_path']}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Get corners and normalize to [0, 1] CAREFULLY
            corners = np.array(item['corners'], dtype=np.float32)
            
            # Validate corners are within image bounds
            corners[:, 0] = np.clip(corners[:, 0], 0, w-1)
            corners[:, 1] = np.clip(corners[:, 1], 0, h-1)
            
            # Normalize coordinates
            normalized_corners = corners.copy()
            normalized_corners[:, 0] /= w
            normalized_corners[:, 1] /= h
            
            # Ensure normalized coordinates are in [0, 1]
            normalized_corners = np.clip(normalized_corners, 0, 1)
            
            # Transform image
            image_tensor = self.base_transform(image)
            
            # Apply conservative photometric augmentation
            if self.augment and random.random() > 0.7:  # Only 30% of the time
                aug_transform = random.choice(self.augment_transforms)
                try:
                    image_tensor = aug_transform(transforms.ToPILImage()(image_tensor))
                    image_tensor = transforms.ToTensor()(image_tensor)
                    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std=[0.229, 0.224, 0.225])(image_tensor)
                except:
                    # If augmentation fails, use original
                    pass
            
            # Flatten corners for output
            corners_flat = normalized_corners.flatten()
            
            return image_tensor, torch.FloatTensor(corners_flat)
            
        except Exception as e:
            logger.error(f"Error loading {item['image_path']}: {e}")
            # Return a dummy sample to avoid breaking the batch
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_corners = torch.zeros(8)
            return dummy_image, dummy_corners

def load_all_available_data():
    """Load ALL available corner annotation data with robust error handling"""
    print("📊 LOADING ALL AVAILABLE CORNER DATA")
    print("=" * 60)
    
    annotation_dirs = [
        ("grey_background_dataset/annotations/train", "grey_background_dataset/images/train"),
        ("grey_background_dataset/annotations/val", "grey_background_dataset/images/val"),
        ("grey_background_dataset/annotations/test", "grey_background_dataset/images/test")
    ]
    
    all_data = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped_files = []
    
    for ann_dir, img_dir in annotation_dirs:
        ann_path = Path(ann_dir)
        img_path = Path(img_dir)
        
        if not ann_path.exists():
            print(f"⚠️  Annotation directory not found: {ann_dir}")
            continue
        if not img_path.exists():
            print(f"⚠️  Image directory not found: {img_dir}")
            continue
        
        split = 'train' if 'train' in ann_dir else 'val' if 'val' in ann_dir else 'test'
        print(f"\n📁 Processing {split} directory: {ann_dir}")
        
        json_files = list(ann_path.glob("*.json"))
        print(f"   Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            # Skip backup files
            if 'backup' in json_file.name.lower():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                if not corners or len(corners) != 4:
                    skipped_files.append(f"{json_file.name} - invalid corners")
                    continue
                
                # Validate corners are reasonable
                corners_array = np.array(corners)
                if np.any(corners_array < 0) or np.any(corners_array > 10000):
                    skipped_files.append(f"{json_file.name} - unreasonable coordinates")
                    continue
                
                # Find corresponding image file
                image_name = annotation.get('image', json_file.stem + '.JPG')
                image_file_path = None
                
                # Try multiple extensions
                for ext in ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']:
                    candidate = img_path / (json_file.stem + ext)
                    if candidate.exists():
                        image_file_path = candidate
                        break
                
                if not image_file_path:
                    skipped_files.append(f"{json_file.name} - image not found")
                    continue
                
                # Validate image can be loaded
                test_image = cv2.imread(str(image_file_path))
                if test_image is None:
                    skipped_files.append(f"{json_file.name} - image load failed")
                    continue
                
                # Add to dataset
                corner_data = {
                    'image_path': str(image_file_path),
                    'corners': corners,
                    'split': split,
                    'annotation_file': str(json_file)
                }
                
                all_data.append(corner_data)
                split_counts[split] += 1
                
            except Exception as e:
                skipped_files.append(f"{json_file.name} - error: {e}")
    
    print(f"\n📊 DATA LOADING SUMMARY:")
    print(f"   Total files loaded: {len(all_data)}")
    print(f"   Train: {split_counts['train']} files")
    print(f"   Val: {split_counts['val']} files")
    print(f"   Test: {split_counts['test']} files")
    print(f"   Skipped: {len(skipped_files)} files")
    
    if skipped_files:
        print(f"\n⚠️  SKIPPED FILES:")
        for skip_reason in skipped_files[:10]:  # Show first 10
            print(f"   {skip_reason}")
        if len(skipped_files) > 10:
            print(f"   ... and {len(skipped_files) - 10} more")
    
    # Create proper train/val split if needed
    train_data = [item for item in all_data if item['split'] == 'train']
    val_data = [item for item in all_data if item['split'] == 'val']
    test_data = [item for item in all_data if item['split'] == 'test']
    
    # If validation set is too small, create one from train
    if len(val_data) < 20 and len(train_data) > 50:
        print(f"\n🔄 REBALANCING SPLITS:")
        print(f"   Val set too small ({len(val_data)}), creating from train set")
        
        # Shuffle and take 20% for validation
        random.shuffle(train_data)
        val_size = max(20, len(train_data) // 5)
        new_val_data = train_data[:val_size]
        new_train_data = train_data[val_size:]
        
        # Update splits
        for item in new_val_data:
            item['split'] = 'val'
        
        train_data = new_train_data
        val_data.extend(new_val_data)
        
        print(f"   New train: {len(train_data)} files")
        print(f"   New val: {len(val_data)} files")
    
    print(f"\n✅ FINAL DATASET:")
    print(f"   Train: {len(train_data)} images")
    print(f"   Val: {len(val_data)} images")
    print(f"   Test: {len(test_data)} images")
    print(f"   Total: {len(train_data) + len(val_data) + len(test_data)} images")
    
    return train_data, val_data, test_data

def train_improved_model():
    """Train improved corner detection model with all available data"""
    print("🚀 TRAINING IMPROVED CORNER DETECTION MODEL")
    print("=" * 60)
    print("Using ALL available training data for maximum performance")
    print()
    
    # Load all data
    train_data, val_data, test_data = load_all_available_data()
    
    if len(train_data) < 50:
        print("❌ Insufficient training data. Need at least 50 samples.")
        return
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    print(f"   Image size: 384x384 (good balance of quality vs speed)")
    print(f"   Backbone: ResNet34 (proven, reliable)")
    print(f"   Augmentation: Conservative (preserve corner relationships)")
    
    # Create datasets
    IMAGE_SIZE = 384  # Good balance between quality and training speed
    train_dataset = RobustCornerDataset(train_data, IMAGE_SIZE, augment=True)
    val_dataset = RobustCornerDataset(val_data, IMAGE_SIZE, augment=False)
    
    # Create data loaders
    batch_size = 16 if len(train_data) > 100 else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    try:
        model = ImprovedCornerModel(backbone='resnet34', pretrained=True)
        print("   ✅ Using ResNet34 backbone")
    except:
        model = ImprovedCornerModel(backbone='resnet18', pretrained=True)
        print("   ⚠️  Fallback to ResNet18 backbone")
    
    model = model.to(device)
    
    # Simple, proven loss function
    criterion = nn.MSELoss()
    
    # Optimizer with proven settings
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
        {'params': model.corner_head.parameters(), 'lr': 1e-3}  # Higher LR for new head
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 40
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    patience = 8
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    pixel_errors = []
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            # Skip dummy samples (from failed loads)
            valid_mask = torch.sum(targets, dim=1) != 0
            if not torch.any(valid_mask):
                continue
            
            images = images[valid_mask].to(device)
            targets = targets[valid_mask].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        pixel_errors_batch = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, targets in val_pbar:
                # Skip dummy samples
                valid_mask = torch.sum(targets, dim=1) != 0
                if not torch.any(valid_mask):
                    continue
                
                images = images[valid_mask].to(device)
                targets = targets[valid_mask].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate pixel error for monitoring
                pred_corners = outputs.cpu().numpy().reshape(-1, 4, 2)
                true_corners = targets.cpu().numpy().reshape(-1, 4, 2)
                
                # Convert back to pixel coordinates
                pred_corners *= IMAGE_SIZE
                true_corners *= IMAGE_SIZE
                
                batch_errors = np.sqrt(np.sum((pred_corners - true_corners) ** 2, axis=2))
                pixel_errors_batch.extend(batch_errors.flatten())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'PixelErr': f'{np.mean(batch_errors):.1f}'
                })
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_pixel_error = np.mean(pixel_errors_batch) if pixel_errors_batch else float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pixel_errors.append(avg_pixel_error)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Avg Pixel Error: {avg_pixel_error:.1f} pixels")
        
        # Save best model based on pixel error (more meaningful than loss)
        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            model_path = "models/improved_corner_detector_all_data.pt"
            Path("models").mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'pixel_error': avg_pixel_error,
                'image_size': IMAGE_SIZE,
                'model_type': 'improved_all_data',
                'train_samples': len(train_data),
                'val_samples': len(val_data)
            }, model_path)
            
            print(f"  ✅ New best model saved! (Pixel Error: {avg_pixel_error:.1f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"  🛑 Early stopping triggered (patience: {patience})")
            break
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != optimizer.param_groups[0]['lr']:
            print(f"  📉 Learning rate reduced to: {current_lr:.2e}")
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print(f"   Best pixel error: {best_pixel_error:.1f} pixels")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model saved to: models/improved_corner_detector_all_data.pt")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(pixel_errors, label='Pixel Error', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Pixels')
    plt.legend()
    plt.title('Validation Pixel Error')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_losses)), [best_pixel_error] * len(train_losses), 
             '--', label=f'Best: {best_pixel_error:.1f}px', color='orange')
    plt.plot(pixel_errors, label='Current', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Pixels')
    plt.legend()
    plt.title('Pixel Error Progress')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_all_data_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"   Training curves saved to: improved_all_data_training_curves.png")
    
    # Expected improvement analysis
    original_error = 64.0  # From previous tests
    improvement = original_error - best_pixel_error
    improvement_pct = (improvement / original_error) * 100
    
    print(f"\n📊 EXPECTED IMPROVEMENT:")
    print(f"   Original model: {original_error:.1f} pixels")
    print(f"   New model: {best_pixel_error:.1f} pixels")
    print(f"   Improvement: {improvement:+.1f} pixels ({improvement_pct:+.1f}%)")
    
    if best_pixel_error < 30:
        print("   🎯 EXCELLENT: Target accuracy achieved!")
    elif best_pixel_error < 50:
        print("   ✅ VERY GOOD: Significant improvement")
    elif best_pixel_error < 60:
        print("   ✅ GOOD: Noticeable improvement")
    else:
        print("   ⚠️  MODEST: Some improvement but more work needed")

def validate_training_data():
    """Validate that we're using all available training data"""
    print("🔍 VALIDATING TRAINING DATA AVAILABILITY")
    print("=" * 60)
    
    # Count all JSON files
    total_json_files = 0
    for ann_dir in ["grey_background_dataset/annotations/train",
                   "grey_background_dataset/annotations/val", 
                   "grey_background_dataset/annotations/test"]:
        ann_path = Path(ann_dir)
        if ann_path.exists():
            json_files = list(ann_path.glob("*.json"))
            non_backup_files = [f for f in json_files if 'backup' not in f.name.lower()]
            total_json_files += len(non_backup_files)
            print(f"📁 {ann_dir}: {len(non_backup_files)} files")
    
    print(f"\n📊 TOTAL AVAILABLE: {total_json_files} annotation files")
    print(f"📊 PREVIOUSLY USED: ~158 files (68% of available)")
    print(f"📊 OPPORTUNITY: +{total_json_files - 158} more files (+{((total_json_files - 158) / 158 * 100):.0f}% more data)")
    
    return total_json_files

def main():
    """Main training function"""
    print("Improved Corner Detection Training - All Data")
    print("=" * 50)
    
    # Validate data availability
    total_files = validate_training_data()
    
    if total_files < 100:
        print("❌ Insufficient data for training")
        return
    
    print(f"\n🎯 IMPROVEMENT STRATEGY:")
    print("✅ Use ALL available training data")
    print("✅ ResNet34 backbone (more capacity than ResNet18)")
    print("✅ Robust data loading with error handling")
    print("✅ Conservative augmentation (preserve corner accuracy)")
    print("✅ Pixel error monitoring (more meaningful than loss)")
    print("✅ Early stopping based on real-world performance")
    
    # Start training
    train_improved_model()

if __name__ == "__main__":
    main()
