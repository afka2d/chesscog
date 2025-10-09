#!/usr/bin/env python3
"""
Train PRECISE corner coordinate regression model (not bounding box detection).
Predicts exact (x,y) coordinates for all 4 corners.
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
from datetime import datetime
import random
from PIL import Image
import pillow_heif

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreciseCornerModel(nn.Module):
    """Regression model that predicts exact corner coordinates"""
    def __init__(self, backbone='resnet18', pretrained=True):
        super(PreciseCornerModel, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Corner regression head - predicts 8 values (4 corners × 2 coordinates)
        self.corner_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8),  # 4 corners * (x, y) normalized to [0, 1]
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

class PreciseCornerDataset(Dataset):
    """Dataset for precise corner coordinate regression"""
    def __init__(self, data, image_size=384, augment=False):
        self.data = data
        self.image_size = image_size
        self.augment = augment
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load image
            image_path = Path(item['image_path'])
            
            if image_path.suffix.lower() in ['.heic']:
                pillow_heif.register_heif_opener()
                img = Image.open(image_path)
                image = np.array(img.convert('RGB'))
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            
            # Get corners and normalize to [0, 1]
            corners = np.array(item['corners'], dtype=np.float32)
            corners[:, 0] = np.clip(corners[:, 0], 0, w-1) / w
            corners[:, 1] = np.clip(corners[:, 1], 0, h-1) / h
            
            # Resize image
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            # Apply transforms
            image_tensor = self.transform(image)
            corners_flat = corners.flatten()  # [x1, y1, x2, y2, x3, y3, x4, y4]
            
            return image_tensor, torch.tensor(corners_flat, dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Error loading {idx}: {e}")
            # Return dummy data
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            dummy_corners = torch.zeros(8)
            return dummy_img, dummy_corners

def load_all_datasets():
    """Load all available corner annotation datasets"""
    all_data = []
    
    # 1. Grey Background
    logger.info("Loading Grey Background dataset...")
    grey_annotations = Path('grey_background_dataset/annotations')
    grey_images = Path('grey_background_dataset/training images')
    
    for split in ['train', 'val', 'test']:
        split_dir = grey_annotations / split
        if split_dir.exists():
            for json_file in split_dir.glob('*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    image_name = data.get('image', json_file.stem + '.JPG')
                    image_path = grey_images / image_name
                    
                    if not image_path.exists():
                        for ext in ['.JPG', '.jpg']:
                            alt = grey_images / (json_file.stem + ext)
                            if alt.exists():
                                image_path = alt
                                break
                    
                    if image_path.exists() and 'corners' in data and len(data['corners']) == 4:
                        all_data.append({
                            'image_path': str(image_path),
                            'corners': data['corners']
                        })
                except Exception as e:
                    pass
    
    logger.info(f"  Loaded {len(all_data)} from Grey Background")
    
    # 2. Marshall Chess
    initial_count = len(all_data)
    logger.info("Loading Marshall Chess dataset...")
    ann_file = Path('marshall_chess_annotations/annotations.json')
    marshall_photos = Path('/Users/tonyblum/Desktop/marshall photos')
    
    if ann_file.exists():
        with open(ann_file) as f:
            data = json.load(f)
        
        for image_name, ann_data in data.get('annotations', {}).items():
            if 'corners' in ann_data and len(ann_data['corners']) == 4:
                image_path = marshall_photos / image_name
                if image_path.exists():
                    all_data.append({
                        'image_path': str(image_path),
                        'corners': ann_data['corners']
                    })
    
    logger.info(f"  Loaded {len(all_data) - initial_count} from Marshall Chess")
    
    # 3. Marshall2
    initial_count = len(all_data)
    logger.info("Loading Marshall2 dataset...")
    ann_dir = Path('marshall2_training_images/annotations')
    img_dir = Path('marshall2_training_images')
    
    if ann_dir.exists():
        for json_file in ann_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if (data.get('annotation_method') == 'manual_interactive' and 
                    'corners' in data and len(data['corners']) == 4):
                    
                    image_name = json_file.stem + '.jpg'
                    image_path = img_dir / image_name
                    
                    if image_path.exists():
                        all_data.append({
                            'image_path': str(image_path),
                            'corners': data['corners']
                        })
            except Exception as e:
                pass
    
    logger.info(f"  Loaded {len(all_data) - initial_count} from Marshall2")
    logger.info(f"\n📊 Total loaded: {len(all_data)} images")
    
    return all_data

def train_precise_corner_model():
    """Train precise corner regression model"""
    print("🚀 TRAINING PRECISE CORNER COORDINATE REGRESSION MODEL")
    print("=" * 60)
    print("This will predict exact (x,y) coordinates for all 4 corners")
    print()
    
    # Load all data
    all_data = load_all_datasets()
    
    if len(all_data) < 100:
        logger.error(f"Insufficient data: {len(all_data)}")
        return
    
    # Split data
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.8)
    val_size = int(len(all_data) * 0.1)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    logger.info(f"\n📊 Data split:")
    logger.info(f"   Train: {len(train_data)}")
    logger.info(f"   Val: {len(val_data)}")
    logger.info(f"   Test: {len(test_data)}")
    
    # Create datasets
    IMAGE_SIZE = 384
    train_dataset = PreciseCornerDataset(train_data, IMAGE_SIZE, augment=False)
    val_dataset = PreciseCornerDataset(val_data, IMAGE_SIZE, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Device: {device}")
    
    model = PreciseCornerModel(backbone='resnet18', pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logger.info(f"\n🎯 Training configuration:")
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   Early stopping patience: {patience}")
    logger.info(f"   Image size: {IMAGE_SIZE}")
    logger.info(f"   Batch size: 16")
    logger.info("\nStarting training...\n")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for images, corners in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, corners = images.to(device), corners.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, corners)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, corners in val_loader:
                images, corners = images.to(device), corners.to(device)
                outputs = model(images)
                loss = criterion(outputs, corners)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate pixel error (convert from normalized)
        train_pixel_error = np.sqrt(train_loss) * IMAGE_SIZE
        val_pixel_error = np.sqrt(val_loss) * IMAGE_SIZE
        
        logger.info(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.6f} ({train_pixel_error:.1f}px), "
                   f"Val Loss={val_loss:.6f} ({val_pixel_error:.1f}px)")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_path = f'models/precise_corner_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"   💾 New best model saved: {save_path} ({val_pixel_error:.1f}px error)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\n⏰ Early stopping triggered at epoch {epoch+1}")
                break
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.6f}")
    logger.info(f"   Best pixel error: {np.sqrt(best_val_loss) * IMAGE_SIZE:.1f} pixels")
    logger.info(f"   Model saved to: {save_path}")

if __name__ == '__main__':
    train_precise_corner_model()


