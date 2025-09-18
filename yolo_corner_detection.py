#!/usr/bin/env python3
"""
YOLO-based corner detection for chessboard detection.
This approach treats the chessboard as an object to detect rather than regressing corner coordinates.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_yolo_installation():
    """Check if YOLOv8 is available"""
    try:
        from ultralytics import YOLO
        print("✅ YOLOv8 (ultralytics) is available")
        return True
    except ImportError:
        print("❌ YOLOv8 not found. Installing...")
        return False

def install_yolo():
    """Install YOLOv8 if not available"""
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✅ YOLOv8 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install YOLOv8: {e}")
        return False

def convert_corners_to_yolo_format():
    """Convert corner annotations to YOLO format for training"""
    print("🔄 CONVERTING CORNER ANNOTATIONS TO YOLO FORMAT")
    print("=" * 60)
    
    # Create YOLO dataset structure
    yolo_dataset_dir = Path("yolo_chessboard_dataset")
    yolo_dataset_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (yolo_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Annotation directories
    annotation_dirs = [
        ("grey_background_dataset/annotations/train", "grey_background_dataset/images/train", "train"),
        ("grey_background_dataset/annotations/val", "grey_background_dataset/images/val", "val"),
        ("grey_background_dataset/annotations/test", "grey_background_dataset/images/test", "test")
    ]
    
    converted_count = 0
    skipped_count = 0
    
    for ann_dir, img_dir, split in annotation_dirs:
        ann_path = Path(ann_dir)
        img_path = Path(img_dir)
        
        if not ann_path.exists() or not img_path.exists():
            continue
        
        print(f"\n📁 Processing {split} split...")
        
        for json_file in ann_path.glob("*.json"):
            if 'backup' in json_file.name.lower():
                continue
            
            try:
                # Load annotation
                with open(json_file, 'r') as f:
                    annotation = json.load(f)
                
                corners = annotation.get('corners', [])
                if not corners or len(corners) != 4:
                    skipped_count += 1
                    continue
                
                # Find corresponding image
                image_name = annotation.get('image', json_file.stem + '.JPG')
                image_file_path = None
                
                for ext in ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']:
                    candidate = img_path / (json_file.stem + ext)
                    if candidate.exists():
                        image_file_path = candidate
                        break
                
                if not image_file_path:
                    skipped_count += 1
                    continue
                
                # Load image to get dimensions
                image = cv2.imread(str(image_file_path))
                if image is None:
                    skipped_count += 1
                    continue
                
                h, w = image.shape[:2]
                
                # Convert corners to YOLO polygon format
                # YOLO expects normalized coordinates (0-1) for polygon
                normalized_corners = []
                for corner in corners:
                    x_norm = corner[0] / w
                    y_norm = corner[1] / h
                    normalized_corners.extend([x_norm, y_norm])
                
                # Create YOLO label (class 0 = chessboard, then polygon coordinates)
                yolo_label = f"0 {' '.join([f'{coord:.6f}' for coord in normalized_corners])}\n"
                
                # Copy image to YOLO dataset
                yolo_image_path = yolo_dataset_dir / split / 'images' / image_file_path.name
                import shutil
                shutil.copy2(image_file_path, yolo_image_path)
                
                # Save YOLO label
                yolo_label_path = yolo_dataset_dir / split / 'labels' / (image_file_path.stem + '.txt')
                with open(yolo_label_path, 'w') as f:
                    f.write(yolo_label)
                
                converted_count += 1
                
            except Exception as e:
                logger.warning(f"Error converting {json_file}: {e}")
                skipped_count += 1
    
    print(f"\n📊 CONVERSION SUMMARY:")
    print(f"   Converted: {converted_count} files")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Dataset location: {yolo_dataset_dir}")
    
    # Create YOLO dataset configuration
    dataset_config = {
        'path': str(yolo_dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes
        'names': ['chessboard']  # Class names
    }
    
    config_path = yolo_dataset_dir / 'dataset.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"   Configuration saved: {config_path}")
    
    return converted_count > 0, yolo_dataset_dir

def train_yolo_corner_model():
    """Train YOLO model for chessboard detection"""
    print("🚀 TRAINING YOLO CHESSBOARD DETECTION MODEL")
    print("=" * 60)
    
    # Check YOLO installation
    if not check_yolo_installation():
        if not install_yolo():
            print("❌ Cannot proceed without YOLO installation")
            return False
    
    # Convert dataset
    success, dataset_dir = convert_corners_to_yolo_format()
    if not success:
        print("❌ Failed to convert dataset")
        return False
    
    try:
        from ultralytics import YOLO
        
        # Initialize YOLO model (start with pre-trained YOLOv8n for speed)
        print("🔧 Initializing YOLOv8 model...")
        model = YOLO('yolov8n-seg.pt')  # Use segmentation model for polygon detection
        
        # Training configuration
        config_path = dataset_dir / 'dataset.yaml'
        
        print("🎯 Starting YOLO training...")
        print(f"   Dataset: {config_path}")
        print(f"   Model: YOLOv8n-seg (segmentation)")
        print(f"   Task: Chessboard polygon detection")
        
        # Train the model
        results = model.train(
            data=str(config_path),
            epochs=50,
            imgsz=640,
            batch=8,
            name='chessboard_detection',
            project='yolo_runs',
            save=True,
            plots=True,
            val=True,
            patience=10,
            device='cpu'  # Use CPU for compatibility
        )
        
        print("✅ YOLO training completed!")
        print(f"   Best model saved to: yolo_runs/chessboard_detection/weights/best.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO training failed: {e}")
        return False

class YOLOCornerDetector:
    """YOLO-based corner detection service"""
    
    def __init__(self, model_path="yolo_runs/chessboard_detection/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"YOLO model not found: {self.model_path}")
                return False
            
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("YOLO chessboard detection model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_corners(self, image_path):
        """
        Detect chessboard corners using YOLO.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if self.model is None:
            logger.error("YOLO model not loaded")
            return None
        
        try:
            # Run YOLO inference
            results = self.model(image_path, verbose=False)
            
            if not results or len(results) == 0:
                logger.warning("No chessboard detected")
                return None
            
            # Get the best detection
            result = results[0]
            
            if result.masks is None or len(result.masks) == 0:
                logger.warning("No segmentation masks found")
                return None
            
            # Get the mask with highest confidence
            best_mask_idx = torch.argmax(result.boxes.conf).item()
            mask = result.masks.xy[best_mask_idx]  # Get polygon coordinates
            
            if len(mask) < 4:
                logger.warning("Insufficient polygon points")
                return None
            
            # Convert polygon to 4 corners
            # For a chessboard, we expect roughly rectangular shape
            corners = self._polygon_to_corners(mask)
            
            return corners.tolist() if corners is not None else None
            
        except Exception as e:
            logger.error(f"YOLO corner detection failed: {e}")
            return None
    
    def _polygon_to_corners(self, polygon_points):
        """Convert polygon points to 4 corners of chessboard"""
        if len(polygon_points) < 4:
            return None
        
        # Find the convex hull to get the outer boundary
        hull = cv2.convexHull(polygon_points.astype(np.float32))
        
        if len(hull) < 4:
            return None
        
        # Approximate polygon to 4 corners
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # If we don't get exactly 4 points, use the 4 most extreme points
        if len(approx) != 4:
            # Find 4 extreme points (top-left, top-right, bottom-right, bottom-left)
            hull_points = hull.reshape(-1, 2)
            
            # Calculate center
            center = np.mean(hull_points, axis=0)
            
            # Find corners by angle
            angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
            
            # Sort by angle and take 4 points at roughly 90-degree intervals
            sorted_indices = np.argsort(angles)
            n_points = len(hull_points)
            
            # Select 4 points evenly distributed around the hull
            corner_indices = [
                sorted_indices[0],
                sorted_indices[n_points // 4],
                sorted_indices[n_points // 2],
                sorted_indices[3 * n_points // 4]
            ]
            
            corners = hull_points[corner_indices]
        else:
            corners = approx.reshape(-1, 2)
        
        # Order corners consistently (top-left, top-right, bottom-right, bottom-left)
        corners = self._order_corners(corners)
        
        return corners
    
    def _order_corners(self, corners):
        """Order corners consistently"""
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort by angle (this gives us a consistent ordering)
        sorted_indices = np.argsort(angles)
        ordered_corners = corners[sorted_indices]
        
        return ordered_corners

def create_yolo_training_script():
    """Create a simplified YOLO training script"""
    print("📝 CREATING YOLO TRAINING SETUP")
    print("=" * 60)
    
    training_script = '''#!/usr/bin/env python3
"""
Simple YOLO training script for chessboard detection.
"""

from ultralytics import YOLO
import yaml

def train_yolo_chessboard():
    """Train YOLO model for chessboard detection"""
    print("🚀 Training YOLO Chessboard Detection")
    
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model
    results = model.train(
        data='yolo_chessboard_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolo_chessboard_v1',
        project='yolo_training_runs',
        save=True,
        plots=True,
        val=True,
        patience=15,
        device='cpu',
        workers=2
    )
    
    print("✅ YOLO training completed!")
    print(f"Best model: yolo_training_runs/yolo_chessboard_v1/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_yolo_chessboard()
'''
    
    with open("train_yolo_chessboard.py", "w") as f:
        f.write(training_script)
    
    print("✅ YOLO training script created: train_yolo_chessboard.py")

def test_yolo_approach():
    """Test YOLO approach for corner detection"""
    print("🧪 TESTING YOLO CORNER DETECTION APPROACH")
    print("=" * 60)
    
    # Check if we can use YOLO
    if not check_yolo_installation():
        if not install_yolo():
            print("❌ Cannot test YOLO without installation")
            return False
    
    # Convert dataset
    print("\n📊 Converting dataset to YOLO format...")
    success, dataset_dir = convert_corners_to_yolo_format()
    
    if success:
        print("✅ Dataset converted successfully")
        
        # Create training script
        create_yolo_training_script()
        
        print(f"\n🎯 YOLO TRAINING SETUP COMPLETE!")
        print(f"   Dataset: {dataset_dir}")
        print(f"   Format: YOLO segmentation (polygon detection)")
        print(f"   Classes: 1 (chessboard)")
        print(f"   Training script: train_yolo_chessboard.py")
        
        print(f"\n🚀 TO START TRAINING:")
        print(f"   python train_yolo_chessboard.py")
        
        print(f"\n💡 YOLO ADVANTAGES:")
        print("   ✅ Purpose-built for object detection")
        print("   ✅ Handles perspective and rotation naturally")
        print("   ✅ Can detect multiple chessboards in one image")
        print("   ✅ Robust to lighting and background variations")
        print("   ✅ Fast inference (real-time capable)")
        print("   ✅ Pre-trained on millions of images")
        
        print(f"\n📊 EXPECTED PERFORMANCE:")
        print("   Target: 20-40 pixel corner accuracy")
        print("   Advantage: Better generalization than CNN regression")
        print("   Speed: Faster inference than current models")
        
        return True
    else:
        print("❌ Failed to convert dataset")
        return False

def create_yolo_vs_cnn_comparison():
    """Create comparison framework for YOLO vs CNN approaches"""
    print(f"\n🔍 YOLO vs CNN CORNER DETECTION COMPARISON")
    print("=" * 60)
    
    comparison = {
        "CNN Regression Approach": {
            "description": "Direct coordinate regression (current approach)",
            "advantages": [
                "Direct corner coordinate output",
                "Smaller model size",
                "Simpler training pipeline",
                "Good for consistent image types"
            ],
            "disadvantages": [
                "Sensitive to perspective changes",
                "Requires precise corner annotations",
                "Limited robustness to scale variations",
                "Prone to overfitting on small datasets"
            ],
            "current_performance": "60-64 pixel average error"
        },
        "YOLO Object Detection": {
            "description": "Treat chessboard as object to detect",
            "advantages": [
                "Robust to perspective and rotation",
                "Pre-trained on massive datasets",
                "Natural handling of scale variations",
                "Can detect multiple boards",
                "Better generalization",
                "Fast inference"
            ],
            "disadvantages": [
                "Requires conversion to polygon format",
                "Larger model size",
                "More complex training pipeline",
                "May need post-processing for exact corners"
            ],
            "expected_performance": "20-40 pixel average error"
        }
    }
    
    for approach, details in comparison.items():
        print(f"\n🎯 {approach.upper()}:")
        print(f"   Description: {details['description']}")
        print(f"   Performance: {details.get('current_performance', details.get('expected_performance'))}")
        
        print("   ✅ Advantages:")
        for advantage in details['advantages']:
            print(f"      • {advantage}")
        
        print("   ⚠️  Disadvantages:")
        for disadvantage in details['disadvantages']:
            print(f"      • {disadvantage}")
    
    print(f"\n💡 RECOMMENDATION:")
    print("   Test both approaches and use the best performing one")
    print("   YOLO may provide better generalization and robustness")

def main():
    """Main function to set up YOLO corner detection"""
    print("YOLO Corner Detection Setup")
    print("=" * 50)
    
    # Test YOLO approach
    success = test_yolo_approach()
    
    if success:
        # Create comparison framework
        create_yolo_vs_cnn_comparison()
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Run: python train_yolo_chessboard.py")
        print("2. Wait for training completion (~30-60 minutes)")
        print("3. Test YOLO model vs current CNN model")
        print("4. Compare accuracy and choose best approach")
        
        print(f"\n📁 FILES CREATED:")
        print("   • yolo_chessboard_dataset/ - YOLO training dataset")
        print("   • train_yolo_chessboard.py - Training script")
        print("   • yolo_corner_detection.py - This setup script")
    else:
        print("❌ YOLO setup failed")

if __name__ == "__main__":
    main()
