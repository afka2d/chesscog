#!/usr/bin/env python3
"""
Enhanced corner detection training with all improvements.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuberLoss(nn.Module):
    """Huber loss for better corner precision (less sensitive to outliers)"""
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()

class GeometricConsistencyLoss(nn.Module):
    """Enforce geometric constraints on corners"""
    def __init__(self, weight=0.1):
        super(GeometricConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, corners):
        # corners: [batch_size, 8] -> [batch_size, 4, 2]
        corners = corners.view(-1, 4, 2)
        
        # Calculate side lengths
        side1 = torch.norm(corners[:, 1] - corners[:, 0], dim=1)  # top
        side2 = torch.norm(corners[:, 2] - corners[:, 1], dim=1)  # right
        side3 = torch.norm(corners[:, 3] - corners[:, 2], dim=1)  # bottom
        side4 = torch.norm(corners[:, 0] - corners[:, 3], dim=1)  # left
        
        # Encourage opposite sides to be similar (rectangular constraint)
        horizontal_consistency = torch.abs(side1 - side3).mean()
        vertical_consistency = torch.abs(side2 - side4).mean()
        
        return self.weight * (horizontal_consistency + vertical_consistency)

class EnhancedCornerModel(nn.Module):
    """Enhanced corner detection model with EfficientNet-B3 backbone"""
    def __init__(self, backbone='efficientnet_b3', pretrained=True):
        super(EnhancedCornerModel, self).__init__()
        
        if backbone == 'efficientnet_b3':
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            # Fallback to ResNet18
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Enhanced corner head with higher capacity
        self.corner_head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8)  # 4 corners * (x, y)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

class EnhancedCornerDataset(Dataset):
    """Enhanced dataset with better augmentations"""
    def __init__(self, data, image_size=512, augment=True):
        self.data = data
        self.image_size = image_size
        self.augment = augment
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms (applied to image and corners together)
        self.augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = cv2.imread(item['image_path'])
        if image is None:
            raise ValueError(f"Could not load image: {item['image_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Get corners and normalize to [0, 1]
        corners = np.array(item['corners'], dtype=np.float32)
        corners[:, 0] /= w  # Normalize x coordinates
        corners[:, 1] /= h  # Normalize y coordinates
        
        # Apply geometric augmentation if enabled
        if self.augment and np.random.rand() > 0.5:
            image, corners = self.apply_geometric_augmentation(image, corners)
        
        # Transform image
        image_tensor = self.base_transform(image)
        
        # Apply photometric augmentation
        if self.augment and np.random.rand() > 0.5:
            image_tensor = self.augment_transform(transforms.ToPILImage()(image_tensor))
            image_tensor = transforms.ToTensor()(image_tensor)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])(image_tensor)
        
        # Flatten corners for output
        corners_flat = corners.flatten()
        
        return image_tensor, torch.FloatTensor(corners_flat)
    
    def apply_geometric_augmentation(self, image, corners):
        """Apply geometric augmentation while maintaining corner consistency"""
        h, w = image.shape[:2]
        
        # Random rotation (small angles to maintain corner validity)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-5, 5)  # Small rotation
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply to image
            image = cv2.warpAffine(image, M, (w, h))
            
            # Apply to corners
            corners_homogeneous = np.column_stack([corners * [w, h], np.ones(4)])
            corners_transformed = M.dot(corners_homogeneous.T).T
            corners = corners_transformed / [w, h]
            
            # Ensure corners are still within bounds
            corners = np.clip(corners, 0, 1)
        
        return image, corners

def load_all_corner_data():
    """Load ALL available corner data (not just subset)"""
    print("📊 Loading ALL available corner data...")
    
    annotation_dirs = [
        ("grey_background_dataset/annotations/train", "grey_background_dataset/images/train"),
        ("grey_background_dataset/annotations/val", "grey_background_dataset/images/val"),
        ("grey_background_dataset/annotations/test", "grey_background_dataset/images/test")
    ]
    
    all_data = []
    train_data = []
    val_data = []
    test_data = []
    
    for ann_dir, img_dir in annotation_dirs:
        ann_path = Path(ann_dir)
        img_path = Path(img_dir)
        
        if not ann_path.exists() or not img_path.exists():
            print(f"⚠️  Skipping {ann_dir} - directory not found")
            continue
        
        split = 'train' if 'train' in ann_dir else 'val' if 'val' in ann_dir else 'test'
        
        for json_file in ann_path.glob("*.json"):
            # Skip backup files
            if 'backup' in json_file.name.lower():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                image_name = annotation.get('image', json_file.stem + '.JPG')
                
                if not corners or len(corners) != 4:
                    print(f"⚠️  Skipping {json_file.name} - invalid corners")
                    continue
                
                # Try multiple image extensions
                image_file_path = None
                for ext in ['.JPG', '.jpg', '.PNG', '.png']:
                    candidate = img_path / (json_file.stem + ext)
                    if candidate.exists():
                        image_file_path = candidate
                        break
                
                if not image_file_path or not image_file_path.exists():
                    print(f"⚠️  Skipping {json_file.name} - image not found")
                    continue
                
                corner_data = {
                    'image_path': str(image_file_path),
                    'corners': corners,
                    'split': split
                }
                
                all_data.append(corner_data)
                
                if split == 'train':
                    train_data.append(corner_data)
                elif split == 'val':
                    val_data.append(corner_data)
                else:
                    test_data.append(corner_data)
                    
            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
    
    # If we have too few validation samples, move some from train
    if len(val_data) < 20 and len(train_data) > 50:
        # Move 20% of train to val
        val_count = max(20, len(train_data) // 5)
        np.random.shuffle(train_data)
        additional_val = train_data[:val_count]
        train_data = train_data[val_count:]
        val_data.extend(additional_val)
        print(f"📊 Moved {len(additional_val)} samples from train to val for better validation")
    
    print(f"📊 Final dataset statistics:")
    print(f"   Total files processed: {len(all_data)}")
    print(f"   Train: {len(train_data)} images")
    print(f"   Val: {len(val_data)} images") 
    print(f"   Test: {len(test_data)} images")
    
    return train_data, val_data, test_data

def train_enhanced_model():
    """Train enhanced corner detection model"""
    print("🚀 TRAINING ENHANCED CORNER DETECTION MODEL")
    print("=" * 60)
    
    # Load all data
    train_data, val_data, test_data = load_all_corner_data()
    
    if len(train_data) < 50:
        print("❌ Insufficient training data. Need at least 50 samples.")
        return
    
    # Create datasets
    IMAGE_SIZE = 512  # Larger image size for better accuracy
    train_dataset = EnhancedCornerDataset(train_data, IMAGE_SIZE, augment=True)
    val_dataset = EnhancedCornerDataset(val_data, IMAGE_SIZE, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    try:
        model = EnhancedCornerModel(backbone='efficientnet_b3', pretrained=True)
        print("✅ Using EfficientNet-B3 backbone")
    except:
        model = EnhancedCornerModel(backbone='resnet18', pretrained=True)  
        print("⚠️  Fallback to ResNet18 backbone")
    
    model = model.to(device)
    
    # Enhanced loss function
    huber_loss = HuberLoss(delta=0.02)  # Small delta for corner precision
    geometric_loss = GeometricConsistencyLoss(weight=0.1)
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower LR for pretrained backbone
        {'params': model.corner_head.parameters(), 'lr': 1e-3}  # Higher LR for corner head
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"🎯 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Combined loss
            primary_loss = huber_loss(outputs, targets)
            geo_loss = geometric_loss(outputs)
            total_loss = primary_loss + geo_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'Huber': f'{primary_loss.item():.6f}',
                'Geo': f'{geo_loss.item():.6f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        pixel_errors = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, targets in val_pbar:
                images, targets = images.to(device), targets.to(device)
                
                outputs = model(images)
                
                primary_loss = huber_loss(outputs, targets)
                geo_loss = geometric_loss(outputs)
                total_loss = primary_loss + geo_loss
                
                val_loss += total_loss.item()
                
                # Calculate pixel error for monitoring
                pred_corners = outputs.cpu().numpy().reshape(-1, 4, 2)
                true_corners = targets.cpu().numpy().reshape(-1, 4, 2)
                
                # Convert back to pixel coordinates (assuming 512x512)
                pred_corners *= IMAGE_SIZE
                true_corners *= IMAGE_SIZE
                
                batch_errors = np.sqrt(np.sum((pred_corners - true_corners) ** 2, axis=2))
                pixel_errors.extend(batch_errors.flatten())
                
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'PixelErr': f'{np.mean(batch_errors):.1f}'
                })
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_pixel_error = np.mean(pixel_errors)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Avg Pixel Error: {avg_pixel_error:.1f} pixels")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            model_path = "models/enhanced_corner_detector_best.pt"
            Path("models").mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'avg_pixel_error': avg_pixel_error,
                'image_size': IMAGE_SIZE,
                'model_type': 'enhanced'
            }, model_path)
            
            print(f"  ✅ New best model saved! (Pixel Error: {avg_pixel_error:.1f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"  🛑 Early stopping triggered (patience: {patience})")
            break
        
        # Learning rate scheduling
        scheduler.step()
    
    print(f"\n🎯 Training completed!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model saved to: models/enhanced_corner_detector_best.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([np.mean(pixel_errors)] * len(train_losses), label='Avg Pixel Error')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Error')
    plt.legend()
    plt.title('Pixel Error Progression')
    
    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"   Training curves saved to: enhanced_training_curves.png")

def main():
    """Main training function"""
    print("Enhanced Corner Detection Training")
    print("=" * 50)
    
    # Check if we have the required data
    if not Path("grey_background_dataset").exists():
        print("❌ grey_background_dataset directory not found!")
        return
    
    train_enhanced_model()

if __name__ == "__main__":
    main()
