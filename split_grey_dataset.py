#!/usr/bin/env python3
"""
Split the images in grey_background_dataset/images into train/val/test subfolders (70/15/15 split).
Move corresponding annotation files as well.
"""
import os
import random
from pathlib import Path
import shutil

IMG_DIR = Path('grey_background_dataset/images')
ANN_DIR = Path('grey_background_dataset/annotations')
SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Get all image files
images = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']]
random.seed(42)
random.shuffle(images)

total = len(images)
train_end = int(total * SPLITS['train'])
val_end = train_end + int(total * SPLITS['val'])

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

# Create split folders and move files
for split, files in splits.items():
    img_split_dir = IMG_DIR / split
    ann_split_dir = ANN_DIR / split
    img_split_dir.mkdir(exist_ok=True)
    ann_split_dir.mkdir(exist_ok=True)
    for img_path in files:
        # Move image
        shutil.move(str(img_path), str(img_split_dir / img_path.name))
        # Move annotation
        ann_name = img_path.with_suffix('.json').name
        ann_path = ANN_DIR / ann_name
        if ann_path.exists():
            shutil.move(str(ann_path), str(ann_split_dir / ann_name))

print('Split complete!') 