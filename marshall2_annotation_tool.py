#!/usr/bin/env python3
"""
Marshall2 Annotation Tool - Modified for marshall2 training images
"""

import cv2
import json
import numpy as np
from pathlib import Path
import chess
from datetime import datetime
import os

def annotate_marshall2_images():
    """Interactive annotation for marshall2 images"""
    
    # Image directory (converted JPG images)
    image_dir = Path("./marshall2_training_images")
    annotations_dir = Path("./marshall2_training_images/annotations")
    visualizations_dir = Path("./marshall2_training_images/visualizations")
    
    # Create directories if they don't exist
    visualizations_dir.mkdir(exist_ok=True)
    
    # Get all images
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")))
    print(f"📁 Found {len(image_files)} marshall2 images to annotate")
    
    if not image_files:
        print("❌ No images found in marshall2_training_images/")
        return
    
    # Load existing progress
    progress_file = Path("./marshall2_training_images/progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "skipped": []}
    
    # Process each image
    for i, image_path in enumerate(image_files):
        # Skip if already completed
        if image_path.name in progress["completed"]:
            print(f"⏭️  Skipping {image_path.name} (already completed)")
            continue
            
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
            
            # Display image for corner annotation
            print("\n🎯 CORNER ANNOTATION:")
            print("Click the 4 corners of the chessboard in this order:")
            print("1. Top-Left (TL)")
            print("2. Top-Right (TR)")  
            print("3. Bottom-Right (BR)")
            print("4. Bottom-Left (BL)")
            print("\nPress 'r' to reset corners, 's' to skip image, 'q' to quit")
            
            corners = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    corners.append([x, y])
                    print(f"📍 Corner {len(corners)}: ({x}, {y})")
                    
                    # Draw corner on image
                    cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(image, (x, y), 12, (255, 255, 255), 2)
                    cv2.putText(image, f'{len(corners)}', (x-10, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw lines between corners
                    if len(corners) >= 2:
                        cv2.line(image, tuple(corners[-2]), tuple(corners[-1]), (255, 0, 0), 2)
                    
                    # Draw complete quadrilateral
                    if len(corners) == 4:
                        corners_np = np.array(corners, dtype=np.int32)
                        cv2.polylines(image, [corners_np.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
                    
                    cv2.imshow('Marshall2 Annotation', image)
            
            # Show image for corner annotation
            cv2.namedWindow('Marshall2 Annotation', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Marshall2 Annotation', mouse_callback)
            cv2.imshow('Marshall2 Annotation', image)
            
            # Wait for 4 corners to be clicked
            while len(corners) < 4:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    # Reset corners
                    corners = []
                    image = cv2.imread(str(image_path))
                    cv2.imshow('Marshall2 Annotation', image)
                    print("🔄 Corners reset")
                elif key == ord('s'):
                    # Skip image
                    cv2.destroyAllWindows()
                    progress["skipped"].append(image_path.name)
                    print("⏭️  Skipping this image")
                    break
                elif key == ord('q'):
                    # Quit
                    cv2.destroyAllWindows()
                    print("🛑 Quitting annotation")
                    return
            
            cv2.destroyAllWindows()
            
            # Check if we skipped
            if image_path.name in progress["skipped"]:
                continue
            
            if len(corners) != 4:
                print("❌ Need exactly 4 corners")
                continue
            
            print(f"✅ Corners captured: {corners}")
            
            # Get FEN from user
            print("\n♟️  FEN ENTRY:")
            print("Enter the chess position in FEN format")
            print("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            print("For starting position, just press Enter (will use default)")
            print("Press 's' to skip this image")
            
            fen_input = input("FEN: ").strip()
            
            if fen_input.lower() == 's':
                progress["skipped"].append(image_path.name)
                print("⏭️  Skipping this image")
                continue
            
            # Use default starting position if empty
            if not fen_input:
                fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                print(f"✅ Using default starting position: {fen_input}")
            
            # Add default ending if not provided
            elif ' ' not in fen_input:
                fen_input += " w KQkq - 0 1"
            
            # Validate FEN
            try:
                chess.Board(fen_input)
                print("✅ FEN is valid")
            except Exception as e:
                print(f"❌ Invalid FEN: {e}")
                print("Please try again or press 's' to skip")
                continue
            
            # Create annotation
            annotation = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "chess_set": "marshall2",
                "corners": corners,
                "fen": fen_input,
                "annotation_method": "manual_interactive",
                "timestamp": datetime.now().isoformat(),
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
            # Save annotation
            annotation_file = annotations_dir / f"{image_path.stem}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"✅ Annotation saved: {annotation_file}")
            
            # Create visualization
            vis_img = cv2.imread(str(image_path))
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
            vis_file = visualizations_dir / f"{image_path.stem}_corners.jpg"
            cv2.imwrite(str(vis_file), vis_img)
            print(f"📊 Visualization saved: {vis_file}")
            
            # Mark as completed
            progress["completed"].append(image_path.name)
            
        except KeyboardInterrupt:
            print("\n🛑 Annotation interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            continue
    
    # Save progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n🎉 Annotation session complete!")
    print(f"📁 Annotations saved in: {annotations_dir}")
    print(f"📊 Visualizations saved in: {visualizations_dir}")
    print(f"✅ Completed: {len(progress['completed'])} images")
    print(f"⏭️  Skipped: {len(progress['skipped'])} images")

if __name__ == "__main__":
    print("🎯 Marshall2 Chess Annotation Tool")
    print("=" * 50)
    print("This tool will help you annotate chess board corners and FEN positions")
    print("for your marshall2 training images.")
    print("\nInstructions:")
    print("1. Click 4 corners of chessboard (TL → TR → BR → BL)")
    print("2. Enter FEN position (or press Enter for starting position)")
    print("3. Press 'r' to reset corners, 's' to skip image")
    print("=" * 50)
    
    annotate_marshall2_images()

