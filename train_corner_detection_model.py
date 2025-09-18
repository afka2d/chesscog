#!/usr/bin/env python3
"""
Train a corner detection model using your existing annotated data.
This creates a completely separate model that won't affect your working API.
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
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessCornerDataset(Dataset):
    def __init__(self, corner_data, transform=None, image_size=512):
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
        
        # Normalize corners to [0, 1] relative to resized image
        corners = np.array(data['corners'], dtype=np.float32)
        
        # Scale corners to match resized image
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        corners[:, 0] *= scale_x  # Scale x coordinates
        corners[:, 1] *= scale_y  # Scale y coordinates
        
        # Normalize to [0, 1]
        corners[:, 0] /= self.image_size
        corners[:, 1] /= self.image_size
        
        # Flatten corners to [x1, y1, x2, y2, x3, y3, x4, y4]
        corners_flat = corners.flatten()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(corners_flat, dtype=torch.float32)

class CornerDetectionModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0'):
        super(CornerDetectionModel, self).__init__()
        
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            # Remove the classifier
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
            # Remove the classifier
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Corner detection head
        self.corner_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8),  # 4 corners × 2 coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

class CornerDetectionTrainer:
    def __init__(self, model_name='corner_detector', image_size=512):
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_corner_data(self):
        """Load corner data from analysis"""
        if not Path("corner_training_info.json").exists():
            print("❌ No corner training info found. Run analyze_corner_data.py first.")
            return None, None, None
        
        # Load existing corner data
        annotation_dirs = [
            "grey_background_dataset/annotations/train",
            "grey_background_dataset/annotations/val", 
            "grey_background_dataset/annotations/test"
        ]
        
        train_data = []
        val_data = []
        test_data = []
        
        for ann_dir in annotation_dirs:
            ann_path = Path(ann_dir)
            if ann_path.exists():
                split = 'train' if 'train' in str(ann_path) else 'val' if 'val' in str(ann_path) else 'test'
                
                for json_file in ann_path.glob("*.json"):
                    if 'backup' in json_file.name:
                        continue
                    
                    try:
                        with open(json_file, 'r') as f:
                            annotation = json.load(f)
                        
                        corners = annotation.get('corners', [])
                        image_name = annotation.get('image', json_file.stem + '.JPG')
                        
                        if corners and len(corners) == 4:
                            # Find corresponding image
                            image_path = self.find_image_path(image_name, ann_path)
                            
                            if image_path and image_path.exists():
                                corner_data = {
                                    'image_path': str(image_path),
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
    
    def find_image_path(self, image_name, ann_dir):
        """Find the corresponding image file"""
        if 'train' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/train")
        elif 'val' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/val")
        elif 'test' in str(ann_dir):
            image_dir = Path("grey_background_dataset/images/test")
        else:
            return None
        
        image_path = image_dir / image_name
        return image_path if image_path.exists() else None
    
    def train_model(self, epochs=50, batch_size=8, learning_rate=0.001):
        """Train the corner detection model"""
        print(f"\n🚀 TRAINING CORNER DETECTION MODEL")
        print("=" * 50)
        
        # Load data
        train_data, val_data, test_data = self.load_corner_data()
        if not train_data:
            print("❌ No training data available")
            return None
        
        # Create datasets
        train_dataset = ChessCornerDataset(train_data, self.train_transform, self.image_size)
        val_dataset = ChessCornerDataset(val_data, self.val_transform, self.image_size)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = CornerDetectionModel(backbone='efficientnet_b0')
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"🎯 Starting training on {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (images, corners) in enumerate(train_loader):
                images, corners = images.to(self.device), corners.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, corners)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, corners in val_loader:
                    images, corners = images.to(self.device), corners.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, corners)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'image_size': self.image_size
                }, f'models/corner_detector_best.pt')
            
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Model saved to: models/corner_detector_best.pt")
        
        # Test the model
        if test_data:
            self.test_model(model, test_data)
        
        return model
    
    def test_model(self, model, test_data):
        """Test the trained model"""
        print(f"\n🧪 TESTING CORNER DETECTION MODEL")
        print("-" * 30)
        
        model.eval()
        test_dataset = ChessCornerDataset(test_data, self.val_transform, self.image_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        total_error = 0.0
        pixel_errors = []
        
        with torch.no_grad():
            for i, (images, corners_gt) in enumerate(test_loader):
                images = images.to(self.device)
                corners_pred = model(images).cpu()
                
                # Convert back to pixel coordinates
                corners_gt_pixels = corners_gt * self.image_size
                corners_pred_pixels = corners_pred * self.image_size
                
                # Calculate pixel error
                pixel_error = torch.mean(torch.sqrt(torch.sum((corners_gt_pixels - corners_pred_pixels) ** 2, dim=1)))
                pixel_errors.append(pixel_error.item())
                total_error += pixel_error.item()
                
                if i < 3:  # Show first 3 examples
                    print(f"   Test {i+1}: Pixel error = {pixel_error.item():.1f} pixels")
        
        avg_pixel_error = total_error / len(test_loader)
        print(f"\n📊 Test Results:")
        print(f"   Average pixel error: {avg_pixel_error:.1f} pixels")
        print(f"   Std pixel error: {np.std(pixel_errors):.1f} pixels")
        
        if avg_pixel_error < 50:
            print("✅ EXCELLENT: Very accurate corner detection")
        elif avg_pixel_error < 100:
            print("✅ GOOD: Acceptable corner detection accuracy")
        elif avg_pixel_error < 200:
            print("⚠️  FAIR: Corner detection needs improvement")
        else:
            print("❌ POOR: Corner detection needs significant work")

def main():
    """Main training function"""
    print("Corner Detection Model Training")
    print("=" * 50)
    print("This will train a model to automatically detect chess board corners")
    print("using your existing annotated data.")
    print()
    print("⚠️  SAFETY NOTE: This creates a completely separate model")
    print("   and will NOT affect your working API in any way.")
    print()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    trainer = CornerDetectionTrainer()
    
    # Training configuration
    epochs = input("Number of epochs? (default: 30): ").strip()
    if not epochs:
        epochs = 30
    else:
        epochs = int(epochs)
    
    batch_size = input("Batch size? (default: 8): ").strip()
    if not batch_size:
        batch_size = 8
    else:
        batch_size = int(batch_size)
    
    print(f"\n🚀 Starting training...")
    print(f"   This will create: models/corner_detector_best.pt")
    print(f"   Your existing API will NOT be affected")
    
    start_time = time.time()
    model = trainer.train_model(epochs=epochs, batch_size=batch_size)
    training_time = time.time() - start_time
    
    if model:
        print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Model saved to: models/corner_detector_best.pt")
        print(f"   Next step: Create corner detection API endpoint")
    else:
        print(f"\n❌ TRAINING FAILED!")
        print(f"   Check your data and try again")

if __name__ == "__main__":
    import os
    main()
