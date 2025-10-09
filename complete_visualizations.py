#!/usr/bin/env python3
"""
Complete visualization generation for marshall2 annotations
"""

import cv2
import json
import numpy as np
from pathlib import Path
import os

def create_remaining_visualizations():
    """Create visualizations for annotations that don't have them yet"""
    
    image_dir = Path("./marshall2_training_images")
    annotations_dir = Path("./marshall2_training_images/annotations")
    visualizations_dir = Path("./marshall2_training_images/visualizations")
    
    # Get all annotation files
    annotation_files = sorted(list(annotations_dir.glob("*.json")))
    print(f"📁 Found {len(annotation_files)} annotation files")
    
    # Get existing visualizations
    existing_viz = set(f.stem.replace("_corners", "") for f in visualizations_dir.glob("*_corners.jpg"))
    print(f"📊 Found {len(existing_viz)} existing visualizations")
    
    # Find missing visualizations
    missing_viz = []
    for ann_file in annotation_files:
        image_stem = ann_file.stem
        if image_stem not in existing_viz:
            missing_viz.append(ann_file)
    
    print(f"🎯 Need to create {len(missing_viz)} missing visualizations")
    
    if not missing_viz:
        print("✅ All visualizations already exist!")
        return
    
    # Create missing visualizations
    for i, ann_file in enumerate(missing_viz):
        print(f"\n📸 Creating visualization {i+1}/{len(missing_viz)}: {ann_file.name}")
        
        try:
            # Load annotation
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # Get image path and corners
            image_path = Path(annotation['image_path'])
            corners = annotation['corners']
            
            if not image_path.exists():
                print(f"❌ Image not found: {image_path}")
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                continue
            
            # Create visualization
            vis_img = image.copy()
            corners_np = np.array(corners, dtype=np.int32)
            
            # Draw corners with labels
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            labels = ['TL', 'TR', 'BR', 'BL']
            
            for j, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
                x, y = corner
                cv2.circle(vis_img, (x, y), 15, color, -1)
                cv2.circle(vis_img, (x, y), 20, (255, 255, 255), 3)
                cv2.putText(vis_img, f'{label}', (x-20, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw quadrilateral
            cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            
            # Save visualization
            vis_file = visualizations_dir / f"{ann_file.stem}_corners.jpg"
            cv2.imwrite(str(vis_file), vis_img)
            print(f"✅ Visualization saved: {vis_file}")
            
        except Exception as e:
            print(f"❌ Error processing {ann_file.name}: {e}")
            continue
    
    print(f"\n🎉 Visualization completion finished!")
    print(f"📊 Total visualizations: {len(list(visualizations_dir.glob('*_corners.jpg')))}")

if __name__ == "__main__":
    print("🎯 Marshall2 Visualization Completion Tool")
    print("=" * 50)
    create_remaining_visualizations()

