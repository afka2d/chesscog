#!/usr/bin/env python3
"""
Investigate the actual differences between Marshall and Original models
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json

def analyze_model_architecture(model_path, model_name):
    """Analyze a model's architecture and parameters"""
    print(f"\n🔍 Analyzing {model_name}")
    print("=" * 50)
    
    try:
        if model_path.suffix == '.pt':
            # Try loading as full model first
            try:
                model = torch.load(str(model_path), map_location='cpu', weights_only=False)
                print(f"✅ Loaded as full model")
                print(f"   Type: {type(model)}")
                print(f"   Architecture: {model.__class__.__name__}")
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"   Total parameters: {total_params:,}")
                print(f"   Trainable parameters: {trainable_params:,}")
                
                # Check if it's a state_dict
                if hasattr(model, 'state_dict'):
                    print(f"   Has state_dict: Yes")
                else:
                    print(f"   Has state_dict: No")
                
                return model
                
            except Exception as e:
                print(f"❌ Failed to load as full model: {e}")
                
                # Try loading as state_dict
                try:
                    state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
                    print(f"✅ Loaded as state_dict")
                    print(f"   Keys: {len(state_dict)}")
                    print(f"   Sample keys: {list(state_dict.keys())[:5]}")
                    
                    # Try to infer architecture from keys
                    if 'fc.weight' in state_dict:
                        print(f"   Likely architecture: ResNet (has fc layer)")
                    elif 'classifier.1.weight' in state_dict:
                        print(f"   Likely architecture: MobileNetV2 (has classifier.1 layer)")
                    elif 'features.0.weight' in state_dict:
                        print(f"   Likely architecture: EfficientNet (has features.0 layer)")
                    else:
                        print(f"   Unknown architecture")
                    
                    return state_dict
                    
                except Exception as e2:
                    print(f"❌ Failed to load as state_dict: {e2}")
                    return None
        else:
            print(f"❌ Not a .pt file: {model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing {model_name}: {e}")
        return None

def compare_models():
    """Compare Marshall vs Original models"""
    print("🧪 Investigating Model Differences")
    print("=" * 60)
    
    # Model paths
    models_to_analyze = [
        # Original models
        ("Original Occupancy", Path("runs/occupancy_classifier/ResNet/ResNet.pt")),
        ("Original Color", Path("models/color_classifier_simple.pt")),
        ("Original Piece", Path("models/piece_classifier_simple.pt")),
        
        # Marshall models
        ("Marshall Occupancy", Path("models_marshall_improved/occupancy_marshall.pt")),
        ("Marshall Color", Path("models_marshall_improved/color_classification_marshall.pt")),
        ("Marshall Piece", Path("models_marshall_improved/piece_classification_combined_marshall.pt")),
    ]
    
    results = {}
    
    for model_name, model_path in models_to_analyze:
        if model_path.exists():
            results[model_name] = analyze_model_architecture(model_path, model_name)
        else:
            print(f"\n❌ {model_name} not found at {model_path}")
            results[model_name] = None
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    
    for model_name, result in results.items():
        if result is not None:
            if isinstance(result, dict):  # state_dict
                print(f"✅ {model_name}: State dict ({len(result)} keys)")
            else:  # full model
                print(f"✅ {model_name}: Full model ({type(result).__name__})")
        else:
            print(f"❌ {model_name}: Not found or failed to load")
    
    return results

if __name__ == "__main__":
    compare_models()
