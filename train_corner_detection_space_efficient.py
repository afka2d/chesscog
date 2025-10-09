#!/usr/bin/env python3
"""
SPACE-EFFICIENT Corner Detection Training Pipeline
Uses symbolic links instead of copying images to save disk space.
"""

import json
import os
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

class SpaceEfficientCornerTrainer:
    """Space-efficient corner detection training using symlinks"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.output_dir = Path(f'yolo_combined_dataset_symlinks_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
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
                    
                    image_name = data.get('image', json_file.stem + '.JPG')
                    image_path = grey_images / image_name
                    
                    if not image_path.exists():
                        for ext in ['.JPG', '.jpg', '.PNG', '.png']:
                            alt_path = grey_images / (json_file.stem + ext)
                            if alt_path.exists():
                                image_path = alt_path
                                break
                    
                    if image_path.exists() and 'corners' in data and len(data['corners']) == 4:
                        self.all_annotations.append({
                            'image_path': str(image_path.absolute()),
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
                    image_path = marshall_photos / image_name
                    
                    if not image_path.exists():
                        jpg_name = image_name.replace('.HEIC', '.jpg').replace('.heic', '.jpg')
                        image_path = marshall_photos / jpg_name
                    
                    if image_path.exists():
                        self.all_annotations.append({
                            'image_path': str(image_path.absolute()),
                            'corners': ann_data['corners'],
                            'dataset': 'marshall_chess',
                            'original_split': None
                        })
                        count += 1
            
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
                
                if (data.get('annotation_method') == 'manual_interactive' and 
                    'corners' in data and len(data['corners']) == 4):
                    
                    image_name = json_file.stem + '.jpg'
                    image_path = img_dir / image_name
                    
                    if image_path.exists():
                        self.all_annotations.append({
                            'image_path': str(image_path.absolute()),
                            'corners': data['corners'],
                            'dataset': 'marshall2',
                            'original_split': None
                        })
                        count += 1
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"✅ Loaded {count} annotations from Marshall2 dataset")
        return count
    
    def convert_to_yolo_format_symlinks(self):
        """Convert to YOLO format using symlinks to save space"""
        logger.info("🔄 Converting annotations to YOLO format (using symlinks)...")
        
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
                    image_path = Path(ann['image_path'])
                    
                    # Get image dimensions (need to read for label creation)
                    if image_path.suffix.lower() in ['.heic']:
                        pillow_heif.register_heif_opener()
                        img = Image.open(image_path)
                        w, h = img.size
                    else:
                        img = cv2.imread(str(image_path))
                        if img is None:
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
                    
                    # Create symlink to image (saves space!)
                    output_image_name = f"{ann['dataset']}_{i:04d}{image_path.suffix}"
                    output_image_path = self.output_dir / split_name / 'images' / output_image_name
                    
                    # Create symbolic link instead of copying
                    if not output_image_path.exists():
                        os.symlink(image_path.absolute(), output_image_path)
                    
                    # Save YOLO label (tiny file)
                    label_name = f"{ann['dataset']}_{i:04d}.txt"
                    label_path = self.output_dir / split_name / 'labels' / label_name
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
        logger.info(f"   💾 Disk space saved: ~10GB (using symlinks)")
        
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
        """Train YOLO model with early stopping"""
        logger.info("🚀 Starting YOLO model training...")
        logger.info("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            # Start from base YOLOv8 detection model
            logger.info("✅ Starting from YOLOv8n base model (detection)")
            model = YOLO('yolov8n.pt')
            
            # Training configuration with safeguards
            training_config = {
                'data': str(dataset_yaml),
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'patience': 15,  # Early stopping
                'save': True,
                'plots': True,
                'val': True,
                'name': f'improved_corner_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'project': 'yolo_training_runs',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': True,
                'device': 'cpu'
            }
            
            logger.info("🎯 Training Configuration:")
            logger.info(f"   Max epochs: {training_config['epochs']}")
            logger.info(f"   Early stopping patience: {training_config['patience']}")
            logger.info(f"   Batch size: {training_config['batch']}")
            logger.info("")
            
            start_time = time.time()
            logger.info(f"🏁 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = model.train(**training_config)
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info(f"✅ Training completed!")
            logger.info(f"⏱️  Total training time: {training_duration/60:.1f} minutes")
            logger.info(f"💾 Best model: {results.save_dir}/weights/best.pt")
            
            return results.save_dir
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def run_full_pipeline(self):
        """Run complete automated training pipeline"""
        logger.info("🚀 SPACE-EFFICIENT CORNER DETECTION TRAINING")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        pipeline_start = time.time()
        
        try:
            # Load all datasets
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
                logger.error(f"❌ Insufficient data: {total_loaded} images")
                return False
            
            # Convert to YOLO format (with symlinks)
            logger.info("STEP 2: Converting to YOLO format (space-efficient)")
            logger.info("-" * 60)
            converted = self.convert_to_yolo_format_symlinks()
            logger.info("")
            
            # Create dataset config
            logger.info("STEP 3: Creating dataset configuration")
            logger.info("-" * 60)
            dataset_yaml = self.create_dataset_yaml()
            logger.info("")
            
            # Train model
            logger.info("STEP 4: Training YOLO model")
            logger.info("-" * 60)
            model_dir = self.train_yolo_model(dataset_yaml)
            logger.info("")
            
            # Final summary
            pipeline_end = time.time()
            total_time = pipeline_end - pipeline_start
            
            logger.info("=" * 60)
            logger.info("🎉 TRAINING PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
            logger.info(f"📊 Training data: {total_loaded} images")
            logger.info(f"💾 New model: {model_dir}/weights/best.pt")
            logger.info(f"📂 Dataset: {self.output_dir}")
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
    print("SPACE-EFFICIENT CORNER DETECTION TRAINING")
    print("=" * 60)
    print()
    print("Uses symbolic links to save disk space!")
    print()
    print("Training data: ~1,400 images")
    print("Disk space needed: <1GB (vs 15GB for full copy)")
    print("Expected time: 1-2 hours")
    print()
    input("Press Enter to start training...")
    print()
    
    trainer = SpaceEfficientCornerTrainer()
    success = trainer.run_full_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
        return 0
    else:
        print("\n❌ Training failed. Check logs for details.")
        return 1

if __name__ == '__main__':
    exit(main())


