#!/usr/bin/env python3
"""
Test script to evaluate the ResNet_uniform model accuracy on the test dataset.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import json
from chesscog.piece_classifier.models import ResNet
from chesscog.core.dataset.dataset import build_dataset
from chesscog.corner_detection.detect_corners import CN

def test_uniform_model_accuracy():
    """Test the ResNet_uniform model accuracy on the test dataset."""
    
    print("=== Testing ResNet_uniform Model Accuracy ===")
    
    # Load configuration
    config = CN.load_yaml_with_base('config/piece_classifier/ResNet_uniform.yaml')
    print(f"Dataset path: {config.DATASET.PATH}")
    print(f"Classes: {config.DATASET.CLASSES}")
    
    # Load the model
    model_path = "runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Model loaded successfully")
    
    # Build test dataset
    from chesscog.core.dataset.dataset import Datasets
    test_dataset = build_dataset(config, Datasets.TEST)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test accuracy
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(len(config.DATASET.CLASSES))}
    class_total = {i: 0 for i in range(len(config.DATASET.CLASSES))}
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            
            total += 1
            if predicted.item() == label.item():
                correct += 1
                class_correct[label.item()] += 1
            class_total[label.item()] += 1
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_dataset)} images...")
    
    # Calculate overall accuracy
    overall_accuracy = correct / total
    print(f"\n=== RESULTS ===")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    
    # Calculate per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, class_name in enumerate(config.DATASET.CLASSES):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name}: No samples")
    
    return overall_accuracy

if __name__ == "__main__":
    accuracy = test_uniform_model_accuracy()
    print(f"\nFinal Model Accuracy: {accuracy:.4f}") 