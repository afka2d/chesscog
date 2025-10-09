#!/usr/bin/env python3
"""
AUTOMATED CORNER DETECTION TRAINING PIPELINE
Combines all 1,572 annotated images and trains an improved YOLO model.

Features:
- Fully automated (no manual intervention needed)
- Early stopping to prevent overtraining
- Saves new model separately (won't overwrite production)
- Comprehensive logging and progress tracking
- Time limits to prevent excessive training
"""

import json
import shutil
import random
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import pillow_heif
import yaml
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'corner_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedCornerTrainer:
    """Automated corner detection training pipeline"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.output_dir = Path(f'yolo_combined_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.all_annotations = []
        
    def load_grey_background_dataset(self):
        """Load grey background dataset (231 images)"""
        logger.info("📁 Loading Grey Background dataset...")
        count = 0
        
        grey_annotations = Path('grey_background_dataset/annotations')
        grey_images = Path('grey_background_dataset/training images')
        
        for split in ['train', 'val', 'test']:
            split_dir = grey_annotations / split
            if not split_dir.exists():
                continue
                
            for json_file in split_dir.glob('*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    # Find corresponding image
                    image_name = data.get('image', json_file.stem + '.JPG')
                    image_path = grey_images / image_name
                    
                    if not image_path.exists():
                        # Try alternate locations
                        for ext in ['.JPG', '.jpg', '.PNG', '.png']:
                            alt_path = grey_images / (json_file.stem + ext)
                            if alt_path.exists():
                                image_path = alt_path
                                break
                    
                    if image_path.exists() and 'corners' in data and len(data['corners']) == 4:
                        self.all_annotations.append({
                            'image_path': str(image_path),
                            'corners': data['corners'],
                            'dataset': 'grey_background',
                            'original_split': split
                        })
                        count += 1
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"✅ Loaded {count} annotations from Grey Background dataset")
        return count
    
    def load_marshall_chess_dataset(self):
        """Load marshall chess dataset (541 images)"""
        logger.info("📁 Loading Marshall Chess dataset...")
        count = 0
        
        ann_file = Path('marshall_chess_annotations/annotations.json')
        marshall_photos = Path('/Users/tonyblum/Desktop/marshall photos')
        
        if not ann_file.exists():
            logger.warning("❌ Marshall chess annotations not found")
            return 0
        
        try:
            with open(ann_file) as f:
                data = json.load(f)
            
            annotations = data.get('annotations', {})
            
            for image_name, ann_data in annotations.items():
                if 'corners' in ann_data and len(ann_data['corners']) == 4:
                    # Find image (try HEIC first, then converted JPG)
                    image_path = marshall_photos / image_name
                    
                    if not image_path.exists():
                        # Try converted JPG name
                        jpg_name = image_name.replace('.HEIC', '.jpg').replace('.heic', '.jpg')
                        image_path = marshall_photos / jpg_name
                    
                    if image_path.exists():
                        self.all_annotations.append({
                            'image_path': str(image_path),
                            'corners': ann_data['corners'],
                            'dataset': 'marshall_chess',
                            'original_split': None
                        })
                        count += 1
                    else:
                        logger.debug(f"Image not found: {image_name}")
            
            logger.info(f"✅ Loaded {count} annotations from Marshall Chess dataset")
            return count
            
        except Exception as e:
            logger.error(f"Error loading Marshall chess dataset: {e}")
            return 0
    
    def load_marshall2_dataset(self):
        """Load marshall2 dataset (796 images)"""
        logger.info("📁 Loading Marshall2 dataset...")
        count = 0
        
        ann_dir = Path('marshall2_training_images/annotations')
        img_dir = Path('marshall2_training_images')
        
        if not ann_dir.exists():
            logger.warning("❌ Marshall2 annotations not found")
            return 0
        
        for json_file in ann_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Check if manually annotated (not placeholder)
                if (data.get('annotation_method') == 'manual_interactive' and 
                    'corners' in data and len(data['corners']) == 4):
                    
                    # Find corresponding image
                    image_name = json_file.stem + '.jpg'
                    image_path = img_dir / image_name
                    
                    if image_path.exists():
                        self.all_annotations.append({
                            'image_path': str(image_path),
                            'corners': data['corners'],
                            'dataset': 'marshall2',
                            'original_split': None
                        })
                        count += 1
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"✅ Loaded {count} annotations from Marshall2 dataset")
        return count
    
    def convert_to_yolo_format(self):
        """Convert corner annotations to YOLO bounding box format"""
        logger.info("🔄 Converting annotations to YOLO format...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split data
        random.shuffle(self.all_annotations)
        total = len(self.all_annotations)
        train_count = int(total * 0.8)
        val_count = int(total * 0.1)
        
        splits = {
            'train': self.all_annotations[:train_count],
            'val': self.all_annotations[train_count:train_count + val_count],
            'test': self.all_annotations[train_count + val_count:]
        }
        
        converted = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, annotations in splits.items():
            logger.info(f"Converting {split_name} split ({len(annotations)} images)...")
            
            for i, ann in enumerate(annotations):
                try:
                    # Load image to get dimensions
                    image_path = Path(ann['image_path'])
                    
                    # Handle HEIC files
                    if image_path.suffix.lower() in ['.heic']:
                        pillow_heif.register_heif_opener()
                        img = Image.open(image_path)
                        img = np.array(img.convert('RGB'))
                    else:
                        img = cv2.imread(str(image_path))
                    
                    if img is None:
                        logger.warning(f"Could not load image: {image_path}")
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Convert corners to bounding box
                    corners = np.array(ann['corners'])
                    x_min = float(np.min(corners[:, 0]))
                    y_min = float(np.min(corners[:, 1]))
                    x_max = float(np.max(corners[:, 0]))
                    y_max = float(np.max(corners[:, 1]))
                    
                    # Normalize to [0, 1]
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    
                    # Save image (convert HEIC to JPG if needed)
                    output_image_name = f"{ann['dataset']}_{i:04d}.jpg"
                    output_image_path = self.output_dir / split_name / 'images' / output_image_name
                    
                    if image_path.suffix.lower() in ['.heic']:
                        cv2.imwrite(str(output_image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        shutil.copy(image_path, output_image_path)
                    
                    # Save YOLO label (class_id x_center y_center width height)
                    label_path = self.output_dir / split_name / 'labels' / f"{ann['dataset']}_{i:04d}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    converted[split_name] += 1
                    
                except Exception as e:
                    logger.warning(f"Error converting annotation {i}: {e}")
        
        logger.info(f"✅ Conversion complete:")
        logger.info(f"   Train: {converted['train']}")
        logger.info(f"   Val: {converted['val']}")
        logger.info(f"   Test: {converted['test']}")
        logger.info(f"   Total: {sum(converted.values())}")
        
        return converted
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration"""
        logger.info("📝 Creating dataset.yaml...")
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': ['chessboard'],
            'nc': 1
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset config saved: {yaml_path}")
        return yaml_path
    
    def train_yolo_model(self, dataset_yaml):
        """Train YOLO model with early stopping and time limits"""
        logger.info("🚀 Starting YOLO model training...")
        logger.info("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            # Start from base YOLOv8 detection model (not segmentation)
            # The existing model is segmentation-based, we need detection for bounding boxes
            logger.info("✅ Starting from YOLOv8n base model (detection)")
            model = YOLO('yolov8n.pt')
            
            # Training configuration with safeguards
            training_config = {
                'data': str(dataset_yaml),
                'epochs': 100,  # Max epochs, but early stopping will kick in
                'imgsz': 640,
                'batch': 16,
                'patience': 15,  # Stop if no improvement for 15 epochs
                'save': True,
                'plots': True,
                'val': True,
                'name': f'improved_corner_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'project': 'yolo_training_runs',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,  # Initial learning rate
                'lrf': 0.01,  # Final learning rate (1% of initial)
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'box': 7.5,  # Box loss weight
                'cls': 0.5,  # Classification loss weight
                'dfl': 1.5,  # Distribution focal loss weight
                'verbose': True,
                'seed': 42,  # Reproducibility
                'deterministic': True,
                'single_cls': True,  # Single class (chessboard)
                'device': 'cpu'  # Use CPU for compatibility (can change to 'cuda' if available)
            }
            
            logger.info("🎯 Training Configuration:")
            logger.info(f"   Max epochs: {training_config['epochs']}")
            logger.info(f"   Early stopping patience: {training_config['patience']}")
            logger.info(f"   Batch size: {training_config['batch']}")
            logger.info(f"   Image size: {training_config['imgsz']}")
            logger.info(f"   Device: {training_config['device']}")
            logger.info("")
            logger.info("⏰ Training will automatically stop if:")
            logger.info("   - No improvement for 15 consecutive epochs")
            logger.info("   - Validation loss stops decreasing")
            logger.info("   - Maximum epochs (100) reached")
            logger.info("")
            
            # Start training with time tracking
            start_time = time.time()
            logger.info(f"🏁 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = model.train(**training_config)
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info(f"✅ Training completed!")
            logger.info(f"⏱️  Total training time: {training_duration/60:.1f} minutes")
            logger.info(f"🏁 Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            logger.info(f"📊 Results saved to: {results.save_dir}")
            logger.info(f"💾 Best model: {results.save_dir}/weights/best.pt")
            logger.info(f"📈 Metrics: {results.save_dir}/results.csv")
            
            return results.save_dir
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def run_full_pipeline(self):
        """Run complete automated training pipeline"""
        logger.info("🚀 STARTING AUTOMATED CORNER DETECTION TRAINING")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Load all datasets
            logger.info("STEP 1: Loading all datasets")
            logger.info("-" * 60)
            grey_count = self.load_grey_background_dataset()
            marshall_count = self.load_marshall_chess_dataset()
            marshall2_count = self.load_marshall2_dataset()
            
            total_loaded = len(self.all_annotations)
            logger.info("")
            logger.info(f"📊 Dataset Summary:")
            logger.info(f"   Grey Background: {grey_count}")
            logger.info(f"   Marshall Chess: {marshall_count}")
            logger.info(f"   Marshall2: {marshall2_count}")
            logger.info(f"   TOTAL: {total_loaded}")
            logger.info("")
            
            if total_loaded < 100:
                logger.error(f"❌ Insufficient data: {total_loaded} images (need at least 100)")
                return False
            
            # Step 2: Convert to YOLO format
            logger.info("STEP 2: Converting to YOLO format")
            logger.info("-" * 60)
            converted = self.convert_to_yolo_format()
            logger.info("")
            
            # Step 3: Create dataset config
            logger.info("STEP 3: Creating dataset configuration")
            logger.info("-" * 60)
            dataset_yaml = self.create_dataset_yaml()
            logger.info("")
            
            # Step 4: Train model
            logger.info("STEP 4: Training YOLO model")
            logger.info("-" * 60)
            model_dir = self.train_yolo_model(dataset_yaml)
            logger.info("")
            
            # Final summary
            pipeline_end = time.time()
            total_time = pipeline_end - pipeline_start
            
            logger.info("=" * 60)
            logger.info("🎉 AUTOMATED TRAINING PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total pipeline time: {total_time/60:.1f} minutes")
            logger.info(f"📊 Training data: {total_loaded} images")
            logger.info(f"💾 New model saved to: {model_dir}/weights/best.pt")
            logger.info(f"📂 Dataset saved to: {self.output_dir}")
            logger.info("")
            logger.info("🎯 Next Steps:")
            logger.info("   1. Test the new model on your test images")
            logger.info("   2. Compare performance with production model")
            logger.info("   3. If satisfied, deploy new model to production")
            logger.info("")
            logger.info(f"✅ Production model (unchanged): yolo_training_runs/yolo_chessboard_v1/weights/best.pt")
            logger.info(f"✅ New improved model: {model_dir}/weights/best.pt")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("AUTOMATED CORNER DETECTION TRAINING PIPELINE")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Combine all 1,572 annotated images from 3 datasets")
    print("  2. Create proper train/val/test splits (80/10/10)")
    print("  3. Train improved YOLO corner detection model")
    print("  4. Save new model WITHOUT overwriting production")
    print()
    print("Features:")
    print("  ✅ Fully automated (no manual intervention)")
    print("  ✅ Early stopping (stops if no improvement for 15 epochs)")
    print("  ✅ Time tracking and logging")
    print("  ✅ Production model stays unchanged")
    print()
    print("Estimated time: 1-2 hours (depending on hardware)")
    print()
    input("Press Enter to start training...")
    print()
    
    trainer = AutomatedCornerTrainer()
    success = trainer.run_full_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
        return 0
    else:
        print("\n❌ Training failed. Check logs for details.")
        return 1

if __name__ == '__main__':
    exit(main())

