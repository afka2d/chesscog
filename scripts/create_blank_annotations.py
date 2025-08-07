import os
import json

# Directory containing the new training images
image_dir = 'grey_background_dataset/training images/'
# Directory to save the annotation files
annotation_dir = 'grey_background_dataset/annotations/train/'

os.makedirs(annotation_dir, exist_ok=True)

# List all JPG images in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

for image_file in image_files:
    annotation = {
        'image': image_file,
        'corners': [],  # To be filled in later
        'fen': ''       # To be filled in later
    }
    base_name = os.path.splitext(image_file)[0]
    annotation_path = os.path.join(annotation_dir, base_name + '.json')
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)

print(f"Created {len(image_files)} blank annotation files in {annotation_dir}") 