#!/usr/bin/env python3
"""
Simple script to test the trained ResNet piece classifier model.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import json

# Load the trained model
model_path = "runs/piece_classifier/ResNet/ResNet.pt"
config_path = "runs/piece_classifier/ResNet/ResNet.yaml"

# Load configuration
with open(config_path, 'r') as f:
    import yaml
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(f"Dataset path: {config.get('DATASET', {}).get('PATH', 'Not found')}")
print(f"Transforms: {config.get('DATASET', {}).get('TRANSFORMS', 'Not found')}")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Define the piece classes (in the order they were trained)
piece_classes = [
    'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
    'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]

# Set up transforms (should match what was used during training)
transform = transforms.Compose([
    transforms.Resize((100, 200)),  # Based on the config
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on a few sample images
test_dir = Path("grey_background_dataset/pieces/test")
print(f"\nTesting on images from: {test_dir}")

total_correct = 0
total_images = 0

for piece_class in piece_classes:
    class_dir = test_dir / piece_class
    if not class_dir.exists():
        continue
    
    images = list(class_dir.glob("*.png"))
    if not images:
        continue
    
    print(f"\nTesting {piece_class} ({len(images)} images):")
    
    for img_path in images[:3]:  # Test first 3 images per class
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_piece = piece_classes[predicted_class]
            is_correct = predicted_piece == piece_class
            
            print(f"  {img_path.name}: Predicted {predicted_piece} (confidence: {confidence:.3f}) - {'✓' if is_correct else '✗'}")
            
            total_correct += int(is_correct)
            total_images += 1
            
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

print(f"\nOverall accuracy: {total_correct}/{total_images} = {total_correct/total_images*100:.1f}%") 