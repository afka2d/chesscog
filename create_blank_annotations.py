#!/usr/bin/env python3
"""
Script to generate blank annotation files for each image in grey_background_dataset/images.
Each annotation will have the same base name as the image and will be placed in grey_background_dataset/annotations.
"""
import os
import json
from pathlib import Path

IMAGES_DIR = 'grey_background_dataset/images'
ANNOTATIONS_DIR = 'grey_background_dataset/annotations'

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    base = Path(img_file).stem
    annotation = {
        "image": img_file,
        "corners": [],
        "fen": "",
        "description": ""
    }
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{base}.json")
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
print(f"Created annotation files for all images in {IMAGES_DIR}") 