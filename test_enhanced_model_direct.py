#!/usr/bin/env python3
"""
Direct test of the enhanced corner detection model.
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import torchvision.transforms as transforms
from enhanced_corner_training import EnhancedCornerModel

def test_enhanced_model_directly():
    """Test the enhanced model directly"""
    print("🎯 TESTING ENHANCED CORNER DETECTION MODEL")
    print("=" * 60)
    
    # Load model
    model_path = "models/enhanced_corner_detector_best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load checkpoint with weights_only=False to handle numpy objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Initialize model
        try:
            model = EnhancedCornerModel(backbone='efficientnet_b3', pretrained=False)
            print("✅ Using EfficientNet-B3 backbone")
        except:
            model = EnhancedCornerModel(backbone='resnet18', pretrained=False)
            print("⚠️  Fallback to ResNet18 backbone")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        image_size = checkpoint.get('image_size', 512)
        print(f"✅ Enhanced model loaded successfully (image size: {image_size})")
        
        # Training stats
        print(f"📊 Model training stats:")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        print(f"   Pixel error: {checkpoint.get('avg_pixel_error', 'N/A'):.1f} pixels")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test on images
    test_images = [
        {
            'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
            'annotation': 'grey_background_dataset/annotations/val/IMG_4779.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4785.JPG', 
            'annotation': 'grey_background_dataset/annotations/test/IMG_4785.json'
        },
        {
            'image': 'grey_background_dataset/images/test/IMG_4763.JPG',
            'annotation': 'grey_background_dataset/annotations/test/IMG_4763.json'
        }
    ]
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    total_error = 0
    valid_tests = 0
    
    for test_case in test_images:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"\n📸 Testing: {Path(image_path).name}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print("   ❌ Could not load image")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Load ground truth
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            gt_corners = np.array(annotation.get('corners', []))
            
            # Preprocess image
            input_tensor = transform(image_rgb).unsqueeze(0).to(device)
            
            # Get predictions
            with torch.no_grad():
                predictions = model(input_tensor)
                predictions = predictions.cpu().numpy().reshape(4, 2)
            
            # Convert normalized coordinates back to pixel coordinates
            pred_corners = predictions * [w, h]
            
            # Calculate errors
            errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
            avg_error = np.mean(errors)
            
            print(f"   📊 Average error: {avg_error:.1f} pixels")
            print(f"   📊 Per-corner errors: {[f'{e:.1f}' for e in errors]} pixels")
            print(f"   📍 Ground truth: {gt_corners.tolist()}")
            print(f"   🤖 Predicted: {pred_corners.astype(int).tolist()}")
            
            total_error += avg_error
            valid_tests += 1
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
    
    # Summary
    if valid_tests > 0:
        overall_avg = total_error / valid_tests
        print(f"\n📊 ENHANCED MODEL PERFORMANCE SUMMARY:")
        print(f"   Valid tests: {valid_tests}")
        print(f"   Overall average error: {overall_avg:.1f} pixels")
        
        # Compare with original performance
        original_avg = 64.0  # From previous tests
        improvement = ((original_avg - overall_avg) / original_avg) * 100
        
        print(f"   Original model average: {original_avg:.1f} pixels")
        print(f"   Improvement: {improvement:.1f}%")
        
        if overall_avg < 30:
            print("   🎯 EXCELLENT: Sub-30 pixel accuracy achieved!")
        elif overall_avg < 50:
            print("   ✅ VERY GOOD: Sub-50 pixel accuracy")
        elif overall_avg < 100:
            print("   ✅ GOOD: Sub-100 pixel accuracy")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Still above 100 pixels")
            
        # Training vs real-world performance
        training_error = checkpoint.get('avg_pixel_error', 160)
        print(f"\n📈 TRAINING vs REAL-WORLD:")
        print(f"   Training error: {training_error:.1f} pixels")
        print(f"   Real-world error: {overall_avg:.1f} pixels")
        
        if overall_avg < training_error * 1.5:
            print("   ✅ Good generalization!")
        else:
            print("   ⚠️  Some overfitting detected")
    else:
        print("❌ No valid tests completed")

def create_improvement_summary():
    """Create a summary of all improvements made"""
    print(f"\n🚀 CORNER DETECTION IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    print("📊 PROBLEM IDENTIFIED:")
    print("• Original accuracy: 64 pixel average error")
    print("• AI corners 'slightly outside' manual corners")
    print("• Only using ~158 of 231+ available training images")
    print("• Basic MSE loss function")
    print("• Simple ResNet18 architecture")
    
    print(f"\n✅ SOLUTIONS IMPLEMENTED:")
    print("1️⃣ DATASET EXPANSION:")
    print("   • Used ALL 231 annotation files (+46% more data)")
    print("   • Better data loading to catch all files")
    print("   • Improved train/val/test splits")
    
    print("2️⃣ ARCHITECTURE IMPROVEMENTS:")
    print("   • Upgraded to EfficientNet-B3 (more powerful)")
    print("   • Enhanced corner head with more capacity")
    print("   • Larger input size (512x512 vs 256x256)")
    
    print("3️⃣ TRAINING IMPROVEMENTS:")
    print("   • Huber Loss (better outlier handling)")
    print("   • Geometric consistency loss")
    print("   • Advanced data augmentation")
    print("   • Different learning rates for backbone vs head")
    print("   • Cosine annealing scheduler")
    
    print("4️⃣ POST-PROCESSING:")
    print("   • Sub-pixel corner refinement (OpenCV)")
    print("   • Geometric validation")
    print("   • Corner order consistency")
    
    print(f"\n🎯 EXPECTED RESULTS:")
    print("• Target: <30 pixel average error (2-3x improvement)")
    print("• Better handling of challenging images")
    print("• More robust to lighting/perspective changes")
    print("• Sub-pixel precision")
    
    print(f"\n📁 KEY FILES CREATED:")
    print("• enhanced_corner_training.py - Complete training pipeline")
    print("• sub_pixel_corner_refinement.py - Refinement service")
    print("• models/enhanced_corner_detector_best.pt - Enhanced model")
    print("• enhanced_training_curves.png - Training visualization")

def main():
    """Main function"""
    print("Enhanced Corner Detection Model - Direct Testing")
    print("=" * 50)
    
    test_enhanced_model_directly()
    create_improvement_summary()

if __name__ == "__main__":
    main()
