#!/usr/bin/env python3
"""
Train a simple piece classifier with proper class balancing and anti-overfitting measures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
import os
from collections import Counter

def get_class_weights(dataset_path):
    """Calculate class weights to handle imbalance."""
    class_counts = []
    class_names = []
    
    for class_dir in Path(dataset_path).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            count = len(list(class_dir.glob("*.png")))
            class_counts.append(count)
            class_names.append(class_name)
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    
    print(f"Class weights: {dict(zip(class_names, class_weights))}")
    return class_weights, class_names

def create_data_loaders():
    """Create balanced data loaders."""
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder("grey_background_dataset/pieces/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder("grey_background_dataset/pieces/val", transform=val_transforms)
    
    # Calculate class weights
    class_weights, class_names = get_class_weights("grey_background_dataset/pieces/train")
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    # Create weighted sampler for training
    sample_weights = [class_weights_tensor[class_idx] for class_idx in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, class_weights_tensor, class_names

def create_simple_model(num_classes):
    """Create a simple ResNet model."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model():
    """Train the model with proper configuration."""
    print("🚀 Training Simple Piece Classifier")
    print("=" * 50)
    
    # Create data loaders
    train_loader, val_loader, class_weights, class_names = create_data_loaders()
    num_classes = len(class_names)
    
    print(f"📊 Classes: {class_names}")
    print(f"📊 Number of classes: {num_classes}")
    
    # Create model
    model = create_simple_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_acc = 0
    
    print(f"\n🏋️  Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
        print(f"   Train Accuracy: {train_acc:.2f}%")
        print(f"   Val Accuracy: {val_acc:.2f}%")
        print(f"   Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Per-class validation accuracy
        print(f"   Per-class Val Accuracy:")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"     {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, "models/piece_classifier/ResNet_simple_balanced.pt")
            print(f"   💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
        print()
    
    print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model

if __name__ == "__main__":
    train_model()
