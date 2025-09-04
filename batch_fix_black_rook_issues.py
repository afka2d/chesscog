#!/usr/bin/env python3
"""
Batch fix script for black rook issues.
This script will systematically fix all source images that generate problematic black rook pieces.
"""

import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path

class BlackRookFixer:
    """Class to handle fixing individual images for black rook issues."""
    
    def __init__(self, image_name, dataset_type):
        self.image_name = image_name
        self.dataset_type = dataset_type
        self.annotation_path = f"grey_background_dataset/annotations/{dataset_type}/{image_name}.json"
        self.image_path = f"grey_background_dataset/images/{dataset_type}/{image_name}.JPG"
        self.pieces_base_dir = f"grey_background_dataset/pieces/{dataset_type}"
        
    def load_current_annotation(self):
        """Load the current annotation file."""
        if not os.path.exists(self.annotation_path):
            return None
        
        with open(self.annotation_path, 'r') as f:
            return json.load(f)
    
    def get_manual_corners(self):
        """Allow user to manually click on the four board corners."""
        print(f"🔍 Loading image: {self.image_path}")
        
        # Load image
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")
        
        # Resize for display if too large
        height, width = image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(image, (new_width, new_height))
            scale_factor = scale
        else:
            display_image = image.copy()
            scale_factor = 1.0
        
        corners = []
        corner_names = ['a8 (top-left)', 'h8 (top-right)', 'h1 (bottom-right)', 'a1 (bottom-left)']
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert display coordinates back to original image coordinates
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                corners.append([orig_x, orig_y])
                
                # Draw the point on the display image
                cv2.circle(display_image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(display_image, f"{len(corners)}", (x+15, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f'Click Board Corners - {self.image_name}', display_image)
                
                print(f"   Corner {len(corners)} ({corner_names[len(corners)-1]}): [{orig_x}, {orig_y}]")
        
        # Create window and set mouse callback
        cv2.namedWindow(f'Click Board Corners - {self.image_name}', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(f'Click Board Corners - {self.image_name}', mouse_callback)
        
        # Display instructions
        print(f"\n📋 CORNER SELECTION INSTRUCTIONS for {self.image_name}:")
        print(f"   Click on the four board corners in this order:")
        print(f"   1. a8 (top-left corner)")
        print(f"   2. h8 (top-right corner)") 
        print(f"   3. h1 (bottom-right corner)")
        print(f"   4. a1 (bottom-left corner)")
        print(f"   Press 'q' to quit, 'r' to reset")
        
        cv2.imshow(f'Click Board Corners - {self.image_name}', display_image)
        
        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("User cancelled corner selection")
            elif key == ord('r'):
                corners.clear()
                display_image = cv2.resize(image, (new_width, new_height)) if scale_factor != 1.0 else image.copy()
                cv2.imshow(f'Click Board Corners - {self.image_name}', display_image)
                print(f"   🔄 Reset - click corners again")
        
        cv2.destroyAllWindows()
        
        # Verify corners form a reasonable rectangle
        corners_np = np.array(corners, dtype=np.float32)
        board_width = max(corners_np[:, 0]) - min(corners_np[:, 0])
        board_height = max(corners_np[:, 1]) - min(corners_np[:, 1])
        aspect_ratio = board_width / board_height
        
        print(f"\n✅ Corner selection complete for {self.image_name}!")
        print(f"   Board dimensions: {board_width:.0f} x {board_height:.0f}")
        print(f"   Aspect ratio: {aspect_ratio:.3f}")
        
        if 0.95 <= aspect_ratio <= 1.05:
            print(f"   🎯 Excellent! Board is nearly square")
        elif 0.9 <= aspect_ratio <= 1.1:
            print(f"   ✅ Good! Board is reasonably square")
        else:
            print(f"   ⚠️  Warning: Board aspect ratio is {aspect_ratio:.3f}")
        
        return corners
    
    def get_manual_fen(self, current_fen):
        """Allow user to manually input the FEN."""
        print(f"\n📝 FEN INPUT for {self.image_name}:")
        print(f"   Current FEN: {current_fen}")
        print(f"   Please verify this FEN matches the actual board position")
        
        while True:
            try:
                fen = input("   Enter correct FEN (or press Enter to keep current): ").strip()
                
                if not fen:  # User wants to keep current FEN
                    fen = current_fen
                    print(f"   ✅ Keeping current FEN: {fen}")
                    break
                
                # Basic FEN validation
                if len(fen.split('/')) == 8:
                    print(f"   ✅ FEN accepted: {fen}")
                    break
                else:
                    print(f"   ❌ Invalid FEN format. Please try again.")
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt("User cancelled FEN input")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return fen
    
    def extract_pieces_with_corners(self, corners, fen):
        """Extract individual piece images using the provided corners and FEN."""
        print(f"\n🔧 Extracting pieces with corrected corners...")
        
        # Load image
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")
        
        # Define target board size for warping
        target_size = 400
        
        # Define target corners (square board)
        target_corners = np.array([
            [0, 0],                    # a8 (top-left)
            [target_size, 0],          # h8 (top-right) 
            [target_size, target_size], # h1 (bottom-right)
            [0, target_size]           # a1 (bottom-left)
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        corners_np = np.array(corners, dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(corners_np, target_corners)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, transform_matrix, (target_size, target_size))
        
        # Save warped board for verification
        debug_dir = "debug_outputs"
        os.makedirs(debug_dir, exist_ok=True)
        warped_path = os.path.join(debug_dir, f"{self.image_name}_warped.png")
        cv2.imwrite(warped_path, warped)
        print(f"   💾 Saved warped board: {warped_path}")
        
        # Parse FEN to get piece positions
        piece_positions = {}
        fen_parts = fen.split()
        board_fen = fen_parts[0]
        
        rank = 8
        file = 0
        
        for char in board_fen:
            if char == '/':
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                square = chr(ord('a') + file) + str(rank)
                piece_type = self.get_piece_type(char)
                piece_positions[square] = piece_type
                file += 1
        
        # Extract individual piece images
        square_size = target_size // 8
        output_dir = f"re_extracted_{self.image_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_count = 0
        for square, piece_type in piece_positions.items():
            if piece_type:  # Skip empty squares
                file_idx = ord(square[0]) - ord('a')
                rank_idx = 8 - int(square[1])
                
                # Calculate square boundaries
                x1 = file_idx * square_size
                y1 = rank_idx * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Extract square
                square_img = warped[y1:y2, x1:x2]
                
                # Save piece image
                filename = f"{self.image_name}_{square}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, square_img)
                extracted_count += 1
        
        print(f"   ✅ Extracted {extracted_count} piece images to: {output_dir}")
        return output_dir, piece_positions
    
    def get_piece_type(self, char):
        """Convert FEN character to piece type."""
        piece_map = {
            'K': 'white_king', 'Q': 'white_queen', 'R': 'white_rook',
            'B': 'white_bishop', 'N': 'white_knight', 'P': 'white_pawn',
            'k': 'black_king', 'q': 'black_queen', 'r': 'black_rook',
            'b': 'black_bishop', 'n': 'black_knight', 'p': 'black_pawn'
        }
        return piece_map.get(char, None)
    
    def update_annotation_file(self, corners, fen):
        """Update the annotation file with corrected corners and FEN."""
        print(f"\n📝 Updating annotation file...")
        
        # Create backup
        backup_path = self.annotation_path + ".backup_before_fix"
        if os.path.exists(self.annotation_path):
            shutil.copy2(self.annotation_path, backup_path)
            print(f"   💾 Created backup: {backup_path}")
        
        # Update annotation
        annotation = {
            "image": f"{self.image_name}.JPG",
            "corners": corners,
            "fen": fen,
            "white_turn": True,
            "timestamp": "corrected_corners_and_fen"
        }
        
        with open(self.annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"   ✅ Updated annotation: {self.annotation_path}")
        print(f"   📊 New corners: {corners}")
        print(f"   📝 New FEN: {fen}")
    
    def replace_dataset_pieces(self, output_dir, piece_positions):
        """Replace old dataset pieces with newly corrected ones."""
        print(f"\n🔧 Replacing dataset pieces...")
        
        if not os.path.exists(output_dir):
            print(f"   ❌ Output directory not found: {output_dir}")
            return 0
        
        piece_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if not piece_files:
            print(f"   ❌ No piece images found in {output_dir}")
            return 0
        
        replaced_count = 0
        backup_count = 0
        
        for piece_file in piece_files:
            # Extract square from filename (e.g., IMG_4752_a2.png -> a2)
            square = piece_file.replace(f'{self.image_name}_', '').replace('.png', '')
            
            if square in piece_positions:
                piece_type = piece_positions[square]
                source_path = os.path.join(output_dir, piece_file)
                
                # Target path in dataset
                target_dir = os.path.join(self.pieces_base_dir, piece_type)
                target_path = os.path.join(target_dir, piece_file)
                
                # Create backup of existing file if it exists
                if os.path.exists(target_path):
                    backup_path = target_path + ".backup_corrected"
                    shutil.copy2(target_path, backup_path)
                    backup_count += 1
                
                # Copy corrected piece to dataset
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(source_path, target_path)
                replaced_count += 1
                
                print(f"   ✅ {piece_file} -> {piece_type}/")
            else:
                print(f"   ⚠️  Unknown square {square} for {piece_file}")
        
        print(f"   📊 Replaced: {replaced_count} pieces")
        print(f"   💾 Backups created: {backup_count} files")
        
        return replaced_count
    
    def cleanup_temp_files(self, output_dir):
        """Clean up temporary files after successful replacement."""
        print(f"\n🧹 Cleaning up temporary files...")
        
        # Remove re-extracted pieces directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"   🗑️  Removed: {output_dir}/")
        
        # Keep warped board images for reference
        print(f"   💾 Kept: debug_outputs/{self.image_name}_warped.png (for reference)")
    
    def fix_image(self):
        """Complete fix process for this image."""
        print(f"\n🔧 FIXING {self.image_name} ({self.dataset_type.upper()} set)")
        print("=" * 60)
        
        try:
            # Step 1: Load current annotation
            annotation = self.load_current_annotation()
            if not annotation:
                print(f"❌ No annotation found for {self.image_name}")
                return False
            
            current_fen = annotation.get('fen', '')
            current_corners = annotation.get('corners', [])
            
            print(f"   📊 Current corners: {current_corners}")
            print(f"   📝 Current FEN: {current_fen}")
            
            # Step 2: Get manual corner coordinates
            print(f"\n🎯 STEP 1: Manual Corner Selection")
            new_corners = self.get_manual_corners()
            
            # Step 3: Get manual FEN
            print(f"\n🎯 STEP 2: FEN Verification")
            new_fen = self.get_manual_fen(current_fen)
            
            # Step 4: Extract pieces with corrected corners
            print(f"\n🎯 STEP 3: Piece Extraction")
            output_dir, piece_positions = self.extract_pieces_with_corners(new_corners, new_fen)
            
            # Step 5: Update annotation file
            print(f"\n🎯 STEP 4: Update Annotation")
            self.update_annotation_file(new_corners, new_fen)
            
            # Step 6: Replace dataset pieces
            print(f"\n🎯 STEP 5: Replace Dataset Pieces")
            replaced_count = self.replace_dataset_pieces(output_dir, piece_positions)
            
            if replaced_count > 0:
                # Step 7: Clean up temporary files
                print(f"\n🎯 STEP 6: Cleanup")
                self.cleanup_temp_files(output_dir)
                
                print(f"\n🎉 {self.image_name} fix complete!")
                print(f"   📊 Total pieces replaced: {replaced_count}")
                return True
            else:
                print(f"\n❌ No pieces were replaced for {self.image_name}")
                return False
                
        except KeyboardInterrupt:
            print(f"\n❌ User cancelled fixing {self.image_name}")
            return False
        except Exception as e:
            print(f"\n❌ Error fixing {self.image_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

def get_black_rook_problematic_images():
    """Get list of all images that need fixing for black rook issues."""
    # Based on the analysis from fix_black_rook_issues.py
    problematic_images = [
        ("NEW_20250805_135338_005", "test"),
        ("IMG_4775", "train"),
        ("NEW_20250805_135337_063", "train"),
        ("IMG_4788", "train"),
        ("NEW_20250805_135338_000", "test"),
        ("IMG_4784", "train"),
        ("IMG_4769", "train"),
        ("NEW_20250805_135337_037", "train"),
        ("NEW_20250805_135337_025", "train"),
        ("NEW_20250805_135338_010", "test"),
        ("NEW_20250805_135337_071", "train"),
        ("NEW_20250805_135338_001", "test"),
        ("NEW_20250805_135338_002", "test"),
        ("NEW_20250805_135338_080", "train"),
        ("NEW_20250805_135338_084", "train"),
        ("IMG_4759", "train"),
        ("NEW_20250805_135338_007", "test"),
        ("NEW_20250805_135337_011", "train"),
        ("NEW_20250805_135337_030", "train"),
        ("NEW_20250805_135338_012", "test"),
        ("NEW_20250805_135337_033", "train"),
        ("NEW_20250805_135338_072", "train"),
        ("NEW_20250805_135338_004", "test"),
        ("NEW_20250805_135338_006", "test"),
        ("IMG_4762", "train"),
        ("NEW_20250805_135338_077", "train"),
        ("NEW_20250805_135337_020", "train"),
        ("IMG_4822", "train"),
        ("NEW_20250805_135338_009", "test"),
    ]
    
    return problematic_images

def update_progress_tracker(image_name, status):
    """Update the progress tracker."""
    tracker_file = "black_rook_fix_progress.json"
    
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            "completed_images": [],
            "in_progress_images": [],
            "failed_images": [],
            "total_images": 29,
            "started_at": None,
            "last_updated": None
        }
    
    import datetime
    
    # Remove from other lists
    if image_name in progress["completed_images"]:
        progress["completed_images"].remove(image_name)
    if image_name in progress["in_progress_images"]:
        progress["in_progress_images"].remove(image_name)
    if image_name in progress["failed_images"]:
        progress["failed_images"].remove(image_name)
    
    # Add to appropriate list
    if status == "completed":
        progress["completed_images"].append(image_name)
    elif status == "in_progress":
        progress["in_progress_images"].append(image_name)
    elif status == "failed":
        progress["failed_images"].append(image_name)
    
    progress["last_updated"] = datetime.datetime.now().isoformat()
    
    with open(tracker_file, 'w') as f:
        json.dump(progress, f, indent=2)

def main():
    """Main function to run batch fixing for black rook issues."""
    print("🔧 Batch Fix for Black Rook Issues")
    print("=" * 60)
    
    # Get list of problematic images
    problematic_images = get_black_rook_problematic_images()
    
    print(f"📊 Found {len(problematic_images)} images to fix:")
    for image_name, dataset_type in problematic_images:
        print(f"   - {image_name} ({dataset_type.upper()})")
    
    print(f"\n💡 This will fix all images that generate problematic black rook pieces.")
    print(f"   Each image will require:")
    print(f"   1. Manual corner selection")
    print(f"   2. FEN verification")
    print(f"   3. Piece regeneration")
    print(f"   4. Dataset replacement")
    
    # Ask user to proceed
    try:
        response = input(f"\n🚀 Proceed with batch fixing? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print(f"❌ Batch fixing cancelled.")
            return
    except KeyboardInterrupt:
        print(f"\n❌ Batch fixing cancelled.")
        return
    
    # Process each image
    successful_fixes = 0
    total_images = len(problematic_images)
    
    for i, (image_name, dataset_type) in enumerate(problematic_images, 1):
        print(f"\n{'='*80}")
        print(f"🔄 PROGRESS: {i}/{total_images}")
        print(f"{'='*80}")
        
        # Update progress tracker
        update_progress_tracker(image_name, "in_progress")
        
        # Create fixer instance
        fixer = BlackRookFixer(image_name, dataset_type)
        
        # Check if image exists
        if not os.path.exists(fixer.image_path):
            print(f"❌ Image not found: {fixer.image_path}")
            print(f"   Skipping to next image...")
            update_progress_tracker(image_name, "failed")
            continue
        
        # Fix the image
        if fixer.fix_image():
            successful_fixes += 1
            update_progress_tracker(image_name, "completed")
            print(f"✅ {image_name} fixed successfully!")
        else:
            update_progress_tracker(image_name, "failed")
            print(f"❌ {image_name} fix failed!")
        
        # Ask if user wants to continue
        if i < total_images:
            try:
                response = input(f"\n🔄 Continue to next image? (y/n): ").strip().lower()
                if response not in ['y', 'yes']:
                    print(f"⏸️  Batch fixing paused. You can resume later.")
                    break
            except KeyboardInterrupt:
                print(f"\n⏸️  Batch fixing paused. You can resume later.")
                break
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 BATCH FIXING COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Results:")
    print(f"   Total images processed: {total_images}")
    print(f"   Successful fixes: {successful_fixes}")
    print(f"   Failed fixes: {total_images - successful_fixes}")
    
    if successful_fixes > 0:
        print(f"\n✅ Black rook issues significantly improved!")
        print(f"   - {successful_fixes} source images fixed")
        print(f"   - All piece types will benefit from these fixes")
        print(f"   - Training data is now more accurate")
    else:
        print(f"\n❌ No images were successfully fixed.")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Check the progress tracker: black_rook_fix_progress.json")
    print(f"   2. Retrain your piece classifier with the improved dataset")
    print(f"   3. Test the new model accuracy")

if __name__ == "__main__":
    main()
