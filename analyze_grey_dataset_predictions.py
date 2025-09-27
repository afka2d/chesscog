#!/usr/bin/env python3
"""
Analyze Grey Dataset Predictions
Show sample images and model predictions to understand poor generalization
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
import matplotlib.pyplot as plt
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GreyDatasetAnalyzer:
    """Analyzer for grey background dataset predictions"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.model_path = Path("models_marshall_improved/piece_classification_combined_marshall.pt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Piece type mapping
        self.piece_types = {
            'P': 0, 'p': 0,  # pawn
            'N': 1, 'n': 1,  # knight
            'B': 2, 'b': 2,  # bishop
            'R': 3, 'r': 3,  # rook
            'Q': 4, 'q': 4,  # queen
            'K': 5, 'k': 5   # king
        }
        
        self.piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        self.num_classes = 6
        
    def load_model(self):
        """Load the combined piece classification model"""
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return None
        
        try:
            # Create model architecture
            model = models.resnet18(weights=None)  # Don't load ImageNet weights
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 6)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"✅ Loaded combined piece classification model from: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def extract_piece_label_from_path(self, img_path):
        """Extract piece type label from image path"""
        try:
            # Try to extract from directory structure
            parts = str(img_path).split('/')
            for part in parts:
                # Handle grey_background_dataset format: black_pawn, white_king, etc.
                if any(piece in part.lower() for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']):
                    piece_name = part.lower()
                    if 'pawn' in piece_name:
                        return 0
                    elif 'knight' in piece_name:
                        return 1
                    elif 'bishop' in piece_name:
                        return 2
                    elif 'rook' in piece_name:
                        return 3
                    elif 'queen' in piece_name:
                        return 4
                    elif 'king' in piece_name:
                        return 5
            
            return None
        except:
            return None
    
    def get_transforms(self):
        """Get data transforms for validation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_grey_dataset_samples(self, num_samples_per_class=5):
        """Load sample images from grey background dataset"""
        logger.info("Loading sample images from grey background dataset...")
        
        samples = []
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
                        # Extract label from directory structure
                        label = self.extract_piece_label_from_path(img_file)
                        if label is not None:
                            # Load image
                            img = Image.open(img_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            samples.append({
                                'image': img,
                                'label': label,
                                'path': str(img_file),
                                'filename': img_file.name
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if samples:
                    break
        
        # Group by class and sample
        class_samples = {i: [] for i in range(self.num_classes)}
        for sample in samples:
            class_samples[sample['label']].append(sample)
        
        # Select samples for each class
        selected_samples = []
        for class_id in range(self.num_classes):
            class_name = self.piece_names[class_id]
            available_samples = class_samples[class_id]
            
            if available_samples:
                # Randomly sample up to num_samples_per_class
                num_to_sample = min(num_samples_per_class, len(available_samples))
                selected = random.sample(available_samples, num_to_sample)
                selected_samples.extend(selected)
                logger.info(f"Selected {len(selected)} {class_name} samples")
            else:
                logger.warning(f"No {class_name} samples found")
        
        logger.info(f"Total selected samples: {len(selected_samples)}")
        return selected_samples
    
    def predict_image(self, model, image):
        """Predict piece type for a single image"""
        transform = self.get_transforms()
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def visualize_predictions(self, model, samples, save_path="grey_dataset_analysis.png"):
        """Visualize sample images with predictions"""
        logger.info("Creating visualization of grey dataset predictions...")
        
        # Calculate grid size
        num_samples = len(samples)
        cols = 5
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample in enumerate(samples):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get prediction
            predicted_class, confidence, all_probs = self.predict_image(model, sample['image'])
            true_class = sample['label']
            
            # Display image
            ax.imshow(sample['image'])
            ax.axis('off')
            
            # Create title with prediction info
            true_name = self.piece_names[true_class]
            pred_name = self.piece_names[predicted_class]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if predicted_class == true_class else 'red'
            
            title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.3f}"
            ax.set_title(title, color=color, fontsize=10)
            
            # Add filename
            ax.text(0.02, 0.98, sample['filename'], transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved: {save_path}")
        plt.close()
    
    def analyze_predictions(self, model, samples):
        """Analyze prediction patterns"""
        logger.info("Analyzing prediction patterns...")
        
        # Count predictions by true class
        class_predictions = {i: {j: 0 for j in range(self.num_classes)} for i in range(self.num_classes)}
        correct_predictions = {i: 0 for i in range(self.num_classes)}
        total_predictions = {i: 0 for i in range(self.num_classes)}
        
        for sample in samples:
            predicted_class, confidence, all_probs = self.predict_image(model, sample['image'])
            true_class = sample['label']
            
            class_predictions[true_class][predicted_class] += 1
            total_predictions[true_class] += 1
            
            if predicted_class == true_class:
                correct_predictions[true_class] += 1
        
        # Print analysis
        logger.info("\n📊 Prediction Analysis:")
        logger.info("=" * 60)
        
        for true_class in range(self.num_classes):
            class_name = self.piece_names[true_class]
            total = total_predictions[true_class]
            correct = correct_predictions[true_class]
            accuracy = (correct / total * 100) if total > 0 else 0
            
            logger.info(f"\n{class_name.upper()} (True Class):")
            logger.info(f"  Total samples: {total}")
            logger.info(f"  Correct predictions: {correct}")
            logger.info(f"  Accuracy: {accuracy:.1f}%")
            
            logger.info("  Predicted as:")
            for pred_class in range(self.num_classes):
                pred_name = self.piece_names[pred_class]
                count = class_predictions[true_class][pred_class]
                percentage = (count / total * 100) if total > 0 else 0
                logger.info(f"    {pred_name}: {count} ({percentage:.1f}%)")
    
    def run_analysis(self):
        """Run complete analysis"""
        logger.info("🚀 Starting Grey Dataset Prediction Analysis")
        logger.info("=" * 60)
        
        # Load model
        model = self.load_model()
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Load sample images
        samples = self.load_grey_dataset_samples(num_samples_per_class=3)
        if not samples:
            logger.error("No samples found")
            return
        
        # Analyze predictions
        self.analyze_predictions(model, samples)
        
        # Create visualization
        self.visualize_predictions(model, samples)
        
        logger.info("\n✅ Analysis completed!")
        logger.info("Check 'grey_dataset_analysis.png' for visual results")

def main():
    """Main analysis function"""
    analyzer = GreyDatasetAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
