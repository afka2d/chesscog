#!/usr/bin/env python3
"""
Resume Marshall2 Annotation Tool - Continue from where you left off
"""

import cv2
import json
import numpy as np
from pathlib import Path
import chess
from datetime import datetime
import os

def resume_marshall2_annotations():
    """Resume annotation for marshall2 images that still need manual annotation"""
    
    # Image directory (converted JPG images)
    image_dir = Path("./marshall2_training_images")
    annotations_dir = Path("./marshall2_training_images/annotations")
    visualizations_dir = Path("./marshall2_training_images/visualizations")
    
    # Create directories if they don't exist
    visualizations_dir.mkdir(exist_ok=True)
    
    # Get all annotation files
    annotation_files = sorted(list(annotations_dir.glob("*.json")))
    
    # Find files that still need annotation (have placeholder coordinates)
    pending_files = []
    completed_files = []
    
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        # Check if this has real manual annotations or placeholder data
        corners = annotation.get('corners', [])
        if len(corners) == 4:
            # Check if coordinates look like placeholders (0,0 to 1000,1000)
            if (corners[0] == [0, 0] and corners[1] == [1000, 0] and 
                corners[2] == [1000, 1000] and corners[3] == [0, 1000]):
                pending_files.append(ann_file)
            else:
                completed_files.append(ann_file)
    
    print(f"📁 Found {len(annotation_files)} total annotation files")
    print(f"✅ Completed: {len(completed_files)} images")
    print(f"⏳ Pending: {len(pending_files)} images")
    
    if not pending_files:
        print("🎉 All images have been manually annotated!")
        return
    
    print(f"\n🎯 Starting annotation from image {len(completed_files) + 1}/{len(annotation_files)}")
    
    # Process each pending image
    for i, ann_file in enumerate(pending_files):
        image_stem = ann_file.stem
        image_path = image_dir / f"{image_stem}.jpg"
        
        print(f"\n{'='*60}")
        print(f"📸 Processing {len(completed_files) + i + 1}/{len(annotation_files)}: {image_path.name}")
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
                    
                    cv2.imshow('Marshall2 Annotation Resume', image)
            
            # Show image for corner annotation
            cv2.namedWindow('Marshall2 Annotation Resume', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Marshall2 Annotation Resume', mouse_callback)
            cv2.imshow('Marshall2 Annotation Resume', image)
            
            # Wait for 4 corners to be clicked
            while len(corners) < 4:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    # Reset corners
                    corners = []
                    image = cv2.imread(str(image_path))
                    cv2.imshow('Marshall2 Annotation Resume', image)
                    print("🔄 Corners reset")
                elif key == ord('s'):
                    # Skip image
                    cv2.destroyAllWindows()
                    print("⏭️  Skipping this image")
                    break
                elif key == ord('q'):
                    # Quit
                    cv2.destroyAllWindows()
                    print("🛑 Quitting annotation")
                    return
            
            cv2.destroyAllWindows()
            
            # Check if we skipped
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
            
            # Update annotation with real data
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
            
            # Save updated annotation
            with open(ann_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"✅ Annotation saved: {ann_file}")
            
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
            vis_file = visualizations_dir / f"{image_stem}_corners.jpg"
            cv2.imwrite(str(vis_file), vis_img)
            print(f"📊 Visualization saved: {vis_file}")
            
        except KeyboardInterrupt:
            print("\n🛑 Annotation interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            continue
    
    print(f"\n🎉 Annotation session complete!")
    print(f"📁 Annotations saved in: {annotations_dir}")
    print(f"📊 Visualizations saved in: {visualizations_dir}")

if __name__ == "__main__":
    print("🎯 Marshall2 Annotation Resume Tool")
    print("=" * 50)
    print("This tool will help you continue annotating chess board corners and FEN positions")
    print("for the remaining marshall2 training images.")
    print("\nInstructions:")
    print("1. Click 4 corners of chessboard (TL → TR → BR → BL)")
    print("2. Enter FEN position (or press Enter for starting position)")
    print("3. Press 'r' to reset corners, 's' to skip image, 'q' to quit")
    print("=" * 50)
    
    resume_marshall2_annotations()

