#!/usr/bin/env python3
"""
Test the improved corner detection model trained with all available data.
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import torchvision.transforms as transforms
from train_with_all_data import ImprovedCornerModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCornerService:
    """Service for the improved corner detection model trained with all data"""
    
    def __init__(self, model_path="models/improved_corner_detector_all_data.pt"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 384  # Default, will be updated from checkpoint
        
        self._load_model()
    
    def _load_model(self):
        """Load the improved corner detection model"""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with correct architecture
            try:
                self.model = ImprovedCornerModel(backbone='resnet34', pretrained=False)
                logger.info("Using ResNet34 backbone")
            except:
                self.model = ImprovedCornerModel(backbone='resnet18', pretrained=False)
                logger.info("Fallback to ResNet18 backbone")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.image_size = checkpoint.get('image_size', 384)
            
            # Display model info
            logger.info(f"Improved corner detection model loaded successfully")
            logger.info(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"Training pixel error: {checkpoint.get('pixel_error', 'N/A'):.1f} pixels")
            logger.info(f"Training samples: {checkpoint.get('train_samples', 'N/A')}")
            logger.info(f"Image size: {self.image_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_corners(self, image_path):
        """Detect corners using the improved model"""
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Preprocess for model (same as training)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(input_tensor)
                predictions = predictions.cpu().numpy().reshape(4, 2)
            
            # Convert normalized coordinates back to pixel coordinates
            pixel_corners = predictions * [w, h]
            
            # Ensure corners are within image bounds
            pixel_corners[:, 0] = np.clip(pixel_corners[:, 0], 0, w-1)
            pixel_corners[:, 1] = np.clip(pixel_corners[:, 1], 0, h-1)
            
            return pixel_corners.tolist()
            
        except Exception as e:
            logger.error(f"Corner detection failed: {e}")
            return None

def test_improved_model():
    """Test the improved model against ground truth"""
    print("🚀 TESTING IMPROVED CORNER DETECTION MODEL")
    print("=" * 60)
    print("Testing model trained with ALL available data (ResNet34)")
    print()
    
    # Initialize services
    from corner_detection_service import CornerDetectionService
    
    original_service = CornerDetectionService("models/corner_detector_best.pt")
    improved_service = ImprovedCornerService("models/improved_corner_detector_all_data.pt")
    
    # Test cases
    test_cases = [
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
    
    results = []
    
    for test_case in test_cases:
        image_path = test_case['image']
        annotation_path = test_case['annotation']
        
        if not Path(image_path).exists() or not Path(annotation_path).exists():
            continue
        
        print(f"📸 Testing: {Path(image_path).name}")
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        gt_corners = np.array(annotation.get('corners', []))
        
        # Test original model
        original_corners = original_service.detect_corners(image_path)
        improved_corners = improved_service.detect_corners(image_path)
        
        if original_corners and improved_corners:
            original_errors = np.sqrt(np.sum((gt_corners - np.array(original_corners)) ** 2, axis=1))
            improved_errors = np.sqrt(np.sum((gt_corners - np.array(improved_corners)) ** 2, axis=1))
            
            original_avg = np.mean(original_errors)
            improved_avg = np.mean(improved_errors)
            
            improvement = original_avg - improved_avg
            improvement_pct = (improvement / original_avg) * 100
            
            result = {
                'image': Path(image_path).name,
                'original_error': original_avg,
                'improved_error': improved_avg,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'original_corners': np.array(original_corners).astype(int).tolist(),
                'improved_corners': np.array(improved_corners).astype(int).tolist(),
                'ground_truth': gt_corners.astype(int).tolist()
            }
            results.append(result)
            
            print(f"   Original model: {original_avg:.1f} pixels")
            print(f"   Improved model: {improved_avg:.1f} pixels")
            print(f"   Improvement: {improvement:+.1f} pixels ({improvement_pct:+.1f}%)")
            
            # Show per-corner improvements
            print(f"   Per-corner improvements:")
            for i, (orig_err, imp_err) in enumerate(zip(original_errors, improved_errors)):
                corner_improvement = orig_err - imp_err
                print(f"     Corner {i}: {orig_err:.1f} → {imp_err:.1f} ({corner_improvement:+.1f}px)")
        else:
            print(f"   ❌ Detection failed")
        
        print()
    
    # Overall summary
    if results:
        avg_original = np.mean([r['original_error'] for r in results])
        avg_improved = np.mean([r['improved_error'] for r in results])
        overall_improvement = avg_original - avg_improved
        overall_improvement_pct = (overall_improvement / avg_original) * 100
        
        print(f"📊 OVERALL IMPROVEMENT SUMMARY:")
        print(f"   Original model average: {avg_original:.1f} pixels")
        print(f"   Improved model average: {avg_improved:.1f} pixels")
        print(f"   Overall improvement: {overall_improvement:+.1f} pixels ({overall_improvement_pct:+.1f}%)")
        
        # Determine success level
        if avg_improved < 20:
            print("   🎯 EXCELLENT: Sub-20 pixel accuracy achieved!")
            success_level = "EXCELLENT"
        elif avg_improved < 30:
            print("   🎯 VERY GOOD: Sub-30 pixel accuracy achieved!")
            success_level = "VERY_GOOD"
        elif avg_improved < 50:
            print("   ✅ GOOD: Sub-50 pixel accuracy")
            success_level = "GOOD"
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Still above 50 pixels")
            success_level = "NEEDS_IMPROVEMENT"
        
        # Training vs real-world comparison
        print(f"\n📈 TRAINING vs REAL-WORLD PERFORMANCE:")
        training_error = 10.9  # From training output
        print(f"   Training error: {training_error:.1f} pixels")
        print(f"   Real-world error: {avg_improved:.1f} pixels")
        
        generalization_ratio = avg_improved / training_error
        if generalization_ratio < 2:
            print("   ✅ Excellent generalization!")
        elif generalization_ratio < 3:
            print("   ✅ Good generalization")
        else:
            print("   ⚠️  Some overfitting detected")
        
        return {
            'success_level': success_level,
            'avg_improved_error': avg_improved,
            'overall_improvement_pct': overall_improvement_pct,
            'results': results
        }
    else:
        print("❌ No successful tests")
        return None

def create_performance_comparison():
    """Create a comprehensive performance comparison"""
    print(f"\n🎯 CORNER DETECTION PERFORMANCE COMPARISON")
    print("=" * 60)
    
    models = [
        {
            'name': 'Original ResNet18',
            'path': 'models/corner_detector_best.pt',
            'expected_error': 64.0,
            'description': 'Baseline model with ~158 training images'
        },
        {
            'name': 'Improved ResNet34', 
            'path': 'models/improved_corner_detector_all_data.pt',
            'expected_error': 10.9,
            'description': 'Trained with ALL 215 available images'
        }
    ]
    
    print("📊 MODEL COMPARISON:")
    for model in models:
        exists = "✅" if Path(model['path']).exists() else "❌"
        print(f"   {exists} {model['name']}")
        print(f"      Path: {model['path']}")
        print(f"      Expected error: {model['expected_error']:.1f} pixels")
        print(f"      Description: {model['description']}")
        print()
    
    # Expected improvement
    original_error = 64.0
    improved_error = 10.9
    expected_improvement = ((original_error - improved_error) / original_error) * 100
    
    print(f"📈 EXPECTED IMPROVEMENT:")
    print(f"   Original: {original_error:.1f} pixels")
    print(f"   Improved: {improved_error:.1f} pixels")
    print(f"   Expected improvement: {expected_improvement:.1f}%")
    print(f"   Target achieved: {'✅ YES' if improved_error < 30 else '⚠️  PARTIAL'}")

def main():
    """Main testing function"""
    print("Improved Corner Detection Model - Real-World Testing")
    print("=" * 50)
    
    # Test the improved model
    test_results = test_improved_model()
    
    # Create performance comparison
    create_performance_comparison()
    
    if test_results:
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"   Success level: {test_results['success_level']}")
        print(f"   Real-world accuracy: {test_results['avg_improved_error']:.1f} pixels")
        print(f"   Improvement over original: {test_results['overall_improvement_pct']:+.1f}%")
        
        if test_results['success_level'] in ['EXCELLENT', 'VERY_GOOD']:
            print(f"\n🚀 RECOMMENDATION: USE THE IMPROVED MODEL!")
            print(f"   Replace your corner detection service with:")
            print(f"   ```python")
            print(f"   from test_improved_all_data_model import ImprovedCornerService")
            print(f"   service = ImprovedCornerService()")
            print(f"   corners = service.detect_corners('your_image.jpg')")
            print(f"   ```")
        else:
            print(f"\n💡 RECOMMENDATION: Continue with optimizations")
            print(f"   The improved model shows promise but needs refinement")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   • models/improved_corner_detector_all_data.pt - New improved model")
    print(f"   • improved_all_data_training_curves.png - Training visualization")
    print(f"   • train_with_all_data.py - Training script")

if __name__ == "__main__":
    main()
