#!/usr/bin/env python3
"""
Scan all dataset images to identify which ones likely have corner coordinate issues.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

def analyze_corner_quality(corners):
    """Analyze corner coordinates for potential issues."""
    if len(corners) != 4:
        return "INVALID", "Wrong number of corners"
    
    # Convert to numpy array
    corners_np = np.array(corners, dtype=np.float32)
    
    # Calculate board dimensions
    board_width = max(corners_np[:, 0]) - min(corners_np[:, 0])
    board_height = max(corners_np[:, 1]) - min(corners_np[:, 1])
    
    # Check aspect ratio (should be close to 1.0 for square board)
    aspect_ratio = board_width / board_height if board_height > 0 else 0
    
    # Check if corners form a reasonable rectangle
    # Calculate distances between adjacent corners
    distances = []
    for i in range(4):
        pt1 = corners_np[i]
        pt2 = corners_np[(i + 1) % 4]
        dist = np.linalg.norm(pt1 - pt2)
        distances.append(dist)
    
    # Check if opposite sides are roughly equal length
    side1_avg = (distances[0] + distances[2]) / 2
    side2_avg = (distances[1] + distances[3]) / 2
    side_ratio = side1_avg / side2_avg if side2_avg > 0 else 0
    
    # Quality assessment
    issues = []
    
    if aspect_ratio < 0.8 or aspect_ratio > 1.25:
        issues.append(f"Poor aspect ratio: {aspect_ratio:.2f}")
    
    if side_ratio < 0.8 or side_ratio > 1.25:
        issues.append(f"Uneven sides: {side_ratio:.2f}")
    
    if board_width < 1000 or board_height < 1000:
        issues.append(f"Board too small: {board_width:.0f}x{board_height:.0f}")
    
    if board_width > 5000 or board_height > 5000:
        issues.append(f"Board too large: {board_width:.0f}x{board_height:.0f}")
    
    # Check for extreme corner positions
    for i, corner in enumerate(corners):
        if corner[0] < 0 or corner[1] < 0:
            issues.append(f"Corner {i} has negative coordinates")
    
    if issues:
        return "PROBLEMATIC", "; ".join(issues)
    else:
        return "GOOD", f"Aspect: {aspect_ratio:.2f}, Sides: {side_ratio:.2f}"

def scan_dataset():
    """Scan the entire dataset for problematic images."""
    print("🔍 Scanning dataset for problematic images...")
    print("=" * 70)
    
    # Dataset structure
    dataset_dirs = [
        "grey_background_dataset/annotations/test",
        "grey_background_dataset/annotations/train", 
        "grey_background_dataset/annotations/val"
    ]
    
    results = {
        'test': {'total': 0, 'good': 0, 'problematic': 0, 'invalid': 0, 'issues': []},
        'train': {'total': 0, 'good': 0, 'problematic': 0, 'invalid': 0, 'issues': []},
        'val': {'total': 0, 'good': 0, 'problematic': 0, 'invalid': 0, 'issues': []}
    }
    
    for dataset_type in ['test', 'train', 'val']:
        print(f"\n🔍 Scanning {dataset_type.upper()} set...")
        
        annotations_dir = f"grey_background_dataset/annotations/{dataset_type}"
        if not os.path.exists(annotations_dir):
            print(f"   ❌ Directory not found: {annotations_dir}")
            continue
        
        # Get all annotation files
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        results[dataset_type]['total'] = len(annotation_files)
        
        print(f"   📁 Found {len(annotation_files)} annotation files")
        
        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            try:
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                
                # Extract image name and corners
                image_name = annotation.get('image', 'unknown')
                corners = annotation.get('corners', [])
                
                # Analyze corner quality
                quality, details = analyze_corner_quality(corners)
                
                if quality == "GOOD":
                    results[dataset_type]['good'] += 1
                elif quality == "PROBLEMATIC":
                    results[dataset_type]['problematic'] += 1
                    results[dataset_type]['issues'].append({
                        'file': annotation_file,
                        'image': image_name,
                        'details': details,
                        'corners': corners
                    })
                else:
                    results[dataset_type]['invalid'] += 1
                
            except Exception as e:
                print(f"   ❌ Error reading {annotation_file}: {e}")
                results[dataset_type]['invalid'] += 1
    
    # Print summary
    print(f"\n📊 DATASET SCAN SUMMARY")
    print("=" * 70)
    
    total_images = 0
    total_problematic = 0
    
    for dataset_type, stats in results.items():
        print(f"\n{dataset_type.upper()}:")
        print(f"   Total images: {stats['total']}")
        print(f"   Good corners: {stats['good']}")
        print(f"   Problematic: {stats['problematic']}")
        print(f"   Invalid: {stats['invalid']}")
        
        total_images += stats['total']
        total_problematic += stats['problematic']
        
        if stats['issues']:
            print(f"   🔍 Sample problematic images:")
            for issue in stats['issues'][:5]:  # Show first 5
                print(f"      - {issue['image']}: {issue['details']}")
            if len(stats['issues']) > 5:
                print(f"      ... and {len(stats['issues']) - 5} more")
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Total images scanned: {total_images}")
    print(f"   Total problematic: {total_problematic}")
    print(f"   Problem rate: {(total_problematic/total_images*100):.1f}%" if total_images > 0 else "N/A")
    
    if total_problematic > 0:
        print(f"\n⚠️  RECOMMENDATIONS:")
        print(f"   1. Fix corner coordinates for problematic images")
        print(f"   2. Verify FEN matches actual board positions")
        print(f"   3. Regenerate piece images with correct coordinates")
        print(f"   4. This will significantly improve classifier accuracy")
    
    return results

def show_problematic_examples():
    """Show examples of problematic images for manual review."""
    print(f"\n🔍 Showing examples of problematic images...")
    
    # Check if we have any problematic images from the scan
    test_issues = []
    train_issues = []
    val_issues = []
    
    # Look for some known problematic images
    known_problematic = [
        "NEW_20250805_135337_020",
        "NEW_20250805_135337_025"
    ]
    
    for image_name in known_problematic:
        # Check test set
        test_annotation = f"grey_background_dataset/annotations/test/{image_name}.json"
        if os.path.exists(test_annotation):
            with open(test_annotation, 'r') as f:
                annotation = json.load(f)
            corners = annotation.get('corners', [])
            quality, details = analyze_corner_quality(corners)
            test_issues.append({
                'image': image_name,
                'quality': quality,
                'details': details,
                'corners': corners
            })
    
    if test_issues:
        print(f"\n🔍 Known problematic images found:")
        for issue in test_issues:
            print(f"   📸 {issue['image']}: {issue['quality']} - {issue['details']}")
            print(f"      Corners: {issue['corners']}")
    
    print(f"\n💡 To fix these images:")
    print(f"   1. Run the corner correction script for each one")
    print(f"   2. Verify the FEN matches the actual board")
    print(f"   3. Regenerate piece images")
    print(f"   4. Replace dataset pieces")

def main():
    """Main function to scan the dataset."""
    print("🔍 Dataset Quality Scanner")
    print("=" * 50)
    
    try:
        # Step 1: Scan entire dataset
        results = scan_dataset()
        
        # Step 2: Show problematic examples
        show_problematic_examples()
        
        print(f"\n✅ Scan complete!")
        
        # Step 3: Provide next steps
        total_problematic = sum(stats['problematic'] for stats in results.values())
        
        if total_problematic > 0:
            print(f"\n🚀 NEXT STEPS:")
            print(f"   1. Focus on fixing the {total_problematic} problematic images")
            print(f"   2. Use the same process we used for NEW_20250805_135338_002")
            print(f"   3. This will dramatically improve your classifier accuracy")
            print(f"   4. Consider fixing them in batches (test set first)")
        else:
            print(f"\n🎉 All images have good corner coordinates!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
