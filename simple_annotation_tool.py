#!/usr/bin/env python3
"""
Simple Interactive Annotation Tool
"""

import cv2
import json
import numpy as np
from pathlib import Path
import requests
import chess
from datetime import datetime
import os

def annotate_images():
    """Simple interactive annotation"""
    
    # Image directory
    image_dir = Path("/Users/tonyblum/Downloads/chess_set2_images")
    output_dir = Path("./chess_set2_annotations")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    # Get all images
    image_files = sorted(list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg")))
    print(f"📁 Found {len(image_files)} images to annotate")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"📸 Processing {i+1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                continue
            
            print(f"✅ Image loaded: {image.shape}")
            
            # Get corners from API
            print("🔍 Detecting corners...")
            with open(image_path, 'rb') as f:
                response = requests.post(
                    "http://localhost:8005/detect_corners",
                    files={'file': f},
                    params={'time_budget': 2.0},
                    timeout=10
                )
            
            if response.status_code != 200:
                print(f"❌ API error: {response.status_code}")
                continue
                
            data = response.json()
            if not data.get('success'):
                print(f"❌ Corner detection failed: {data}")
                continue
                
            corners = data['corners']
            print(f"✅ Corners detected: {corners}")
            
            # Show image with corners
            vis_img = image.copy()
            corners_np = np.array(corners, dtype=np.int32)
            
            # Draw corners
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
            vis_file = output_dir / "visualizations" / f"{image_path.stem}_corners.jpg"
            cv2.imwrite(str(vis_file), vis_img)
            print(f"📊 Visualization saved: {vis_file}")
            
            # Get FEN from user
            print("\n♟️  Enter FEN position:")
            print("Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            print("Or press Enter to skip this image")
            
            fen_input = input("FEN: ").strip()
            
            if not fen_input:
                print("⏭️  Skipping this image")
                continue
            
            # Add default ending if not provided
            if ' ' not in fen_input:
                fen_input += " w KQkq - 0 1"
            
            # Validate FEN
            try:
                chess.Board(fen_input)
                print("✅ FEN is valid")
            except Exception as e:
                print(f"❌ Invalid FEN: {e}")
                print("Please try again or press Enter to skip")
                continue
            
            # Create annotation
            annotation = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "chess_set": "set2",
                "corners": corners,
                "fen": fen_input,
                "annotation_method": "simple_interactive",
                "corner_detection_api": "robust_port_8005",
                "timestamp": datetime.now().isoformat()
            }
            
            # Save annotation
            annotation_file = output_dir / "annotations" / f"{image_path.stem}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"✅ Annotation saved: {annotation_file}")
            
        except KeyboardInterrupt:
            print("\n🛑 Annotation interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            continue
    
    print(f"\n🎉 Annotation complete!")
    print(f"📁 Annotations saved in: {output_dir / 'annotations'}")
    print(f"📊 Visualizations saved in: {output_dir / 'visualizations'}")

if __name__ == "__main__":
    annotate_images()
