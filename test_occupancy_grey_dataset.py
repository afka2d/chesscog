#!/usr/bin/env python3
"""
Test Marshall Occupancy Model on Grey Background Dataset
Check if occupancy model has the same generalization issues as piece classification
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OccupancyDataset(Dataset):
    """Dataset for occupancy detection validation"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        square = item['square']
        label = item['label']
        
        # Handle different data formats
        if isinstance(square, torch.Tensor):
            # Already a tensor (from previous data)
            square_pil = transforms.ToPILImage()(square)
        else:
            # BGR image (from Marshall data)
            if len(square.shape) == 3 and square.shape[2] == 3:
                square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square_pil = Image.fromarray(square)
        
        if self.transform:
            square_pil = self.transform(square_pil)
        
        return square_pil, label

class OccupancyGreyTester:
    """Tester for Marshall occupancy model on grey background dataset"""
    
    def __init__(self):
        """Initialize the tester"""
        self.model_path = Path("models_marshall_improved/occupancy_marshall.pt")
        self.original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Occupancy labels: 0 = empty, 1 = occupied
        self.labels = ['empty', 'occupied']
        self.num_classes = 2
        
    def load_marshall_occupancy_model(self):
        """Load the Marshall occupancy model"""
        if not self.model_path.exists():
            logger.error(f"Marshall occupancy model not found: {self.model_path}")
            return None
        
        if not self.original_model_path.exists():
            logger.error(f"Original occupancy model not found: {self.original_model_path}")
            return None
        
        try:
            # Load the original model architecture
            model = torch.load(str(self.original_model_path), map_location='cpu', weights_only=False)
            logger.info("✅ Original model architecture loaded")
            
            # Load the Marshall weights (state_dict)
            marshall_weights = torch.load(str(self.model_path), map_location='cpu', weights_only=True)
            logger.info("✅ Marshall weights loaded")
            
            # Apply the Marshall weights to the original model architecture
            model.load_state_dict(marshall_weights)
            logger.info("✅ Marshall weights applied to model")
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading Marshall occupancy model: {e}")
            return None
    
    def create_grey_occupancy_dataset(self):
        """Create occupancy dataset from grey background pieces"""
        logger.info("Creating occupancy dataset from grey background pieces...")
        
        dataset = []
        data_paths = [
            "grey_background_dataset/pieces/train",
            "grey_background_dataset/pieces/val", 
            "grey_background_dataset/pieces/test"
        ]
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                logger.info(f"Found data at: {path}")
                
                # Load images and labels
                for img_file in path.rglob("*.png"):
                    try:
                        # All pieces in grey dataset are occupied squares
                        label = 1  # occupied
                        
                        # Load image
                        img = Image.open(img_file)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Convert to tensor
                        transform = transforms.Compose([
                            transforms.Resize((100, 100)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_tensor = transform(img)
                        
                        dataset.append({
                            'square': img_tensor,
                            'label': label,
                            'source': 'grey_background',
                            'image_name': img_file.name
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if dataset:
                    break
        
        logger.info(f"Loaded {len(dataset)} grey background occupancy samples")
        return dataset
    
    def create_empty_squares_dataset(self):
        """Create empty squares dataset for comparison"""
        logger.info("Creating empty squares dataset...")
        
        # We'll create synthetic empty squares by taking random patches
        # from the grey background and treating them as empty
        dataset = []
        data_paths = [
            "grey_background_dataset/pieces/train",
            "grey_background_dataset/pieces/val", 
            "grey_background_dataset/pieces/test"
        ]
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                # Get a few sample images to create empty squares from
                sample_files = list(path.rglob("*.png"))[:10]  # Take first 10 files
                
                for img_file in sample_files:
                    try:
                        # Load image
                        img = Image.open(img_file)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Convert to numpy array
                        img_array = np.array(img)
                        
                        # Create multiple empty square samples by taking random patches
                        for _ in range(5):  # Create 5 empty samples per image
                            # Take a random 100x100 patch
                            h, w = img_array.shape[:2]
                            if h >= 100 and w >= 100:
                                y = np.random.randint(0, h - 100)
                                x = np.random.randint(0, w - 100)
                                patch = img_array[y:y+100, x:x+100]
                                
                                # Convert back to PIL
                                patch_pil = Image.fromarray(patch)
                                
                                # Convert to tensor
                                transform = transforms.Compose([
                                    transforms.Resize((100, 100)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
                                img_tensor = transform(patch_pil)
                                
                                dataset.append({
                                    'square': img_tensor,
                                    'label': 0,  # empty
                                    'source': 'synthetic_empty',
                                    'image_name': f"empty_{img_file.name}"
                                })
                        
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if dataset:
                    break
        
        logger.info(f"Created {len(dataset)} synthetic empty square samples")
        return dataset
    
    def get_transforms(self):
        """Get data transforms for validation"""
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_model(self, model, dataset, dataset_name):
        """Evaluate model on a dataset"""
        logger.info(f"\n🔍 Evaluating Marshall occupancy model on {dataset_name}...")
        
        if len(dataset) == 0:
            logger.warning(f"No data available for {dataset_name}")
            return None
        
        # Create data loader
        val_dataset = OccupancyDataset(dataset, self.get_transforms())
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Evaluate
        all_predictions = []
        all_labels = []
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Calculate overall accuracy
        correct = sum(class_correct)
        total = sum(class_total)
        overall_acc = 100.0 * correct / total
        
        logger.info(f"📊 {dataset_name} Results:")
        logger.info(f"Overall accuracy: {overall_acc:.2f}% ({correct}/{total})")
        
        # Per-class accuracy
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"{self.labels[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Classification report
        logger.info(f"\n📋 Detailed Classification Report for {dataset_name}:")
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.labels, 
                                     digits=3)
        logger.info(report)
        
        return {
            'overall_accuracy': overall_acc,
            'class_accuracies': [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                               for i in range(self.num_classes)],
            'class_totals': class_total,
            'class_correct': class_correct,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def create_confusion_matrix(self, results, dataset_name):
        """Create and save confusion matrix"""
        if results is None:
            return
        
        # Create confusion matrix
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.labels, 
                   yticklabels=self.labels)
        plt.title(f'Occupancy Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        output_path = f"occupancy_confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Confusion matrix saved: {output_path}")
        plt.close()
    
    def run_test(self):
        """Run complete test"""
        logger.info("🚀 Testing Marshall Occupancy Model on Grey Background Dataset")
        logger.info("=" * 70)
        
        # Load model
        model = self.load_marshall_occupancy_model()
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Create datasets
        occupied_data = self.create_grey_occupancy_dataset()
        empty_data = self.create_empty_squares_dataset()
        
        # Combine datasets
        combined_data = occupied_data + empty_data
        
        # Evaluate on combined dataset
        results = self.evaluate_model(model, combined_data, "Grey Background Dataset")
        if results:
            self.create_confusion_matrix(results, "Grey Background Dataset")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 SUMMARY")
        logger.info("=" * 70)
        
        if results:
            logger.info(f"Grey Background Dataset Accuracy: {results['overall_accuracy']:.2f}%")
            logger.info(f"Empty squares accuracy: {results['class_accuracies'][0]:.2f}%")
            logger.info(f"Occupied squares accuracy: {results['class_accuracies'][1]:.2f}%")
            
            if results['overall_accuracy'] > 80:
                logger.info("✅ Good generalization - model works well on grey background data")
            elif results['overall_accuracy'] > 60:
                logger.info("⚠️ Moderate generalization - some performance degradation")
            else:
                logger.info("❌ Poor generalization - significant performance degradation")
        else:
            logger.info("❌ No results available")
        
        logger.info("\n✅ Test completed!")

def main():
    """Main test function"""
    tester = OccupancyGreyTester()
    tester.run_test()

if __name__ == "__main__":
    main()
