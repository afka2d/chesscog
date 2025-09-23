#!/usr/bin/env python3
"""
Test the piece classifier models using the same evaluation approach as training.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
import os

def test_model_with_evaluation_approach():
    """Test the model using the same approach as the evaluation system."""
    print("🔍 Testing Model with Evaluation Approach")
    print("=" * 50)
    
    # Load the ResNet_uniform model
    model_path = "runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📦 Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Define transforms exactly as in training config
    transforms_test = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset_path = "grey_background_dataset/pieces/test"
    if not os.path.exists(test_dataset_path):
        print(f"❌ Test dataset not found: {test_dataset_path}")
        return
    
    print(f"📁 Loading test dataset from {test_dataset_path}")
    test_dataset = ImageFolder(test_dataset_path, transform=transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get class names
    class_names = test_dataset.classes
    print(f"📋 Classes: {class_names}")
    
    # Test the model
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    predictions_by_class = {}
    
    print(f"\n🧪 Testing with {len(test_dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                
                # Track predictions by class
                predicted_class = class_names[predicted[i]]
                true_class = class_names[label]
                if true_class not in predictions_by_class:
                    predictions_by_class[true_class] = []
                predictions_by_class[true_class].append(predicted_class)
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx * 32}/{len(test_dataset)} images...")
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f"\n📊 Results:")
    print(f"   Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    
    # Per-class accuracy
    print(f"\n📈 Per-class Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"   {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Check for knight bias
    knight_predictions = 0
    total_predictions = 0
    for true_class, preds in predictions_by_class.items():
        for pred in preds:
            total_predictions += 1
            if 'knight' in pred:
                knight_predictions += 1
    
    knight_percentage = (knight_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n🔍 Knight Analysis:")
    print(f"   Knight predictions: {knight_predictions}/{total_predictions} ({knight_percentage:.1f}%)")
    
    if knight_percentage > 50:
        print("   🚨 WARNING: Model has strong knight bias!")
    else:
        print("   ✅ Knight distribution looks normal")
    
    # Show prediction distribution
    print(f"\n📊 Prediction Distribution:")
    pred_counts = {}
    for true_class, preds in predictions_by_class.items():
        for pred in preds:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    for pred, count in sorted(pred_counts.items()):
        percentage = (count / total_predictions) * 100
        print(f"   {pred}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    test_model_with_evaluation_approach()
