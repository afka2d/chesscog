import os
import json
import subprocess
from pathlib import Path

image_dir = 'grey_background_dataset/training images/'
annotation_dir = 'grey_background_dataset/annotations/train/'

# List all JPG images
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])

for image_file in image_files:
    base_name = Path(image_file).stem
    annotation_path = os.path.join(annotation_dir, base_name + '.json')
    # Check if annotation exists and has corners
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            if data.get('corners') and len(data['corners']) == 4:
                print(f"✅ Skipping {image_file} (corners already annotated)")
                continue
    # Run the interactive picker
    print(f"\n➡️  Annotating corners for {image_file}")
    img_path = os.path.join(image_dir, image_file)
    subprocess.run(['python', 'interactive_corner_picker.py', img_path])
    # After picker closes, check if corners were saved
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            if data.get('corners') and len(data['corners']) == 4:
                print(f"✅ Corners saved for {image_file}")
            else:
                print(f"⚠️  Corners not saved for {image_file}, please try again.")
    else:
        print(f"❌ Annotation file not found for {image_file}") 