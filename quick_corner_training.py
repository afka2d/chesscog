#!/usr/bin/env python3
"""
Quick corner detection training - lighter version for faster training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import json
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickCornerDataset(Dataset):
    def __init__(self, corner_data, transform=None, image_size=256):
        self.corner_data = corner_data
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.corner_data)
    
    def __getitem__(self, idx):
        data = self.corner_data[idx]
        
        # Load image
        image = cv2.imread(data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Scale and normalize corners
        corners = np.array(data['corners'], dtype=np.float32)
        
        # Scale corners to match resized image
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        
        # Normalize to [0, 1]
        corners[:, 0] /= self.image_size
        corners[:, 1] /= self.image_size
        
        # Flatten corners
        corners_flat = corners.flatten()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(corners_flat, dtype=torch.float32)

class LightweightCornerModel(nn.Module):
    def __init__(self):
        super(LightweightCornerModel, self).__init__()
        
        # Use a lightweight ResNet18
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        
        # Simple corner detection head
        self.corner_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),  # 4 corners × 2 coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

def load_corner_data():
    """Load corner data from annotations"""
    annotation_dirs = [
        ("grey_background_dataset/annotations/train", "grey_background_dataset/images/train"),
        ("grey_background_dataset/annotations/val", "grey_background_dataset/images/val"),
        ("grey_background_dataset/annotations/test", "grey_background_dataset/images/test")
    ]
    
    train_data = []
    val_data = []
    test_data = []
    
    for ann_dir, img_dir in annotation_dirs:
        ann_path = Path(ann_dir)
        img_path = Path(img_dir)
        
        if not ann_path.exists() or not img_path.exists():
            continue
        
        split = 'train' if 'train' in ann_dir else 'val' if 'val' in ann_dir else 'test'
        
        for json_file in ann_path.glob("*.json"):
            if 'backup' in json_file.name:
                continue
            
            try:
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                image_name = annotation.get('image', json_file.stem + '.JPG')
                
                if corners and len(corners) == 4:
                    image_file_path = img_path / image_name
                    
                    if image_file_path.exists():
                        corner_data = {
                            'image_path': str(image_file_path),
                            'corners': corners
                        }
                        
                        if split == 'train':
                            train_data.append(corner_data)
                        elif split == 'val':
                            val_data.append(corner_data)
                        else:
                            test_data.append(corner_data)
                            
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
    
    print(f"📊 Loaded corner data:")
    print(f"   Train: {len(train_data)} images")
    print(f"   Val: {len(val_data)} images")
    print(f"   Test: {len(test_data)} images")
    
    return train_data, val_data, test_data

def quick_train_corner_model():
    """Quick training of corner detection model"""
    print("🚀 QUICK CORNER DETECTION TRAINING")
    print("=" * 50)
    print("This will train a lightweight corner detection model")
    print("without affecting your main API.")
    print()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_data, val_data, test_data = load_corner_data()
    
    if len(train_data) < 10:
        print("❌ Insufficient training data")
        return False
    
    # Create datasets
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = QuickCornerDataset(train_data, transform, image_size=256)
    val_dataset = QuickCornerDataset(val_data, transform, image_size=256) if val_data else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) if val_dataset else None
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightCornerModel().to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20  # Quick training
    best_val_loss = float('inf')
    
    print(f"🎯 Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, corners) in enumerate(train_loader):
            images, corners = images.to(device), corners.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, corners)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        val_loss = 0.0
        if val_loader:
            model.eval()
            val_batches = 0
            
            with torch.no_grad():
                for images, corners in val_loader:
                    images, corners = images.to(device), corners.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, corners)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        else:
            avg_val_loss = avg_train_loss
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'image_size': 256
            }, 'models/corner_detector_best.pt')
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model saved to: models/corner_detector_best.pt")
    
    return True

def main():
    """Main function"""
    print("Quick Corner Detection Training")
    print("=" * 50)
    
    success = quick_train_corner_model()
    
    if success:
        print(f"\n🎯 TRAINING SUCCESSFUL!")
        print("Next steps:")
        print("1. Start corner detection API: python corner_detection_api.py")
        print("2. Test the system: python test_corner_detection.py")
        print("3. View demo at: http://localhost:8002/demo")
    else:
        print(f"\n❌ TRAINING FAILED!")

if __name__ == "__main__":
    main()
