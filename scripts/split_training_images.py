import os
import shutil
import random
from pathlib import Path

# Parameters
image_dir = Path('grey_background_dataset/training images')
annotation_dir = Path('grey_background_dataset/annotations/train')
output_image_dir = Path('grey_background_dataset/images')
output_ann_dir = Path('grey_background_dataset/annotations')
splits = ['train', 'val', 'test']
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

# Gather all image files
image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
image_files.sort()
random.seed(42)
random.shuffle(image_files)

total = len(image_files)
n_train = int(total * train_ratio)
n_val = int(total * val_ratio)
train_files = image_files[:n_train]
val_files = image_files[n_train:n_train+n_val]
test_files = image_files[n_train+n_val:]
split_map = {'train': train_files, 'val': val_files, 'test': test_files}

for split in splits:
    (output_image_dir / split).mkdir(parents=True, exist_ok=True)
    (output_ann_dir / split).mkdir(parents=True, exist_ok=True)
    for img_path in split_map[split]:
        # Copy image
        shutil.copy2(img_path, output_image_dir / split / img_path.name)
        # Copy annotation
        ann_name = img_path.stem + '.json'
        ann_path = annotation_dir / ann_name
        dest_ann_path = output_ann_dir / split / ann_name
        if ann_path.exists():
            if ann_path.resolve() != dest_ann_path.resolve():
                shutil.copy2(ann_path, dest_ann_path)
        else:
            print(f"Warning: Annotation not found for {img_path.name}")

print(f"Split {total} images: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.") 