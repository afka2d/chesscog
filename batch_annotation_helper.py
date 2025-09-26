#!/usr/bin/env python3
"""
Batch Annotation Helper
======================

Helper script for batch processing chess images with semi-automated annotation.
Includes quality control and progress tracking.

Features:
- Batch processing with progress tracking
- Quality control checks
- Resume capability
- Statistics and reporting
- Integration with robust corner detection API
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import requests
import chess
from typing import List, Dict, Optional
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchAnnotationHelper:
    """
    Helper for batch annotation processing
    """
    
    def __init__(self, images_dir: str, output_dir: str, chess_set: str = "set2"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.chess_set = chess_set
        self.corner_api_url = "http://localhost:8005"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Load image files
        self.image_files = self._load_image_files()
        self.progress_file = self.output_dir / "progress.json"
        self.stats_file = self.output_dir / "statistics.json"
        
        # Load progress if exists
        self.progress = self._load_progress()
        
        logger.info(f"📁 Loaded {len(self.image_files)} images")
        logger.info(f"📊 Progress: {self.progress['completed']}/{len(self.image_files)} completed")
    
    def _load_image_files(self) -> List[Path]:
        """Load all image files"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
        
        return sorted(image_files)
    
    def _load_progress(self) -> Dict:
        """Load progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "last_processed": None,
            "start_time": datetime.now().isoformat(),
            "completed_files": [],
            "skipped_files": [],
            "failed_files": []
        }
    
    def _save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _get_auto_corners(self, image_path: str) -> Optional[List[List[float]]]:
        """Get automatic corner detection"""
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f"{self.corner_api_url}/detect_corners",
                    files={'file': f},
                    params={'time_budget': 2.0},
                    timeout=10
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['corners']
            return None
        except Exception as e:
            logger.error(f"Auto-detection failed for {image_path}: {e}")
            return None
    
    def _validate_fen(self, fen: str) -> bool:
        """Validate FEN string"""
        try:
            chess.Board(fen)
            return True
        except:
            return False
    
    def _create_visualization(self, image_path: str, corners: List[List[float]], 
                            status: str = "Auto-detected") -> str:
        """Create visualization image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        vis_img = image.copy()
        corners_np = np.array(corners, dtype=np.int32)
        
        # Draw corners
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
            x, y = corner
            cv2.circle(vis_img, (x, y), 15, color, -1)
            cv2.circle(vis_img, (x, y), 20, (255, 255, 255), 3)
            cv2.putText(vis_img, f'{label}', (x-20, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw quadrilateral
        cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        
        # Add status text
        cv2.putText(vis_img, f"{status} - {Path(image_path).name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(vis_img, f"{status} - {Path(image_path).name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Save visualization
        vis_file = self.output_dir / "visualizations" / f"{Path(image_path).stem}_corners.jpg"
        cv2.imwrite(str(vis_file), vis_img)
        return str(vis_file)
    
    def process_single_image(self, image_path: Path, auto_corners: Optional[List[List[float]]] = None) -> Dict:
        """
        Process a single image with auto-detection and manual input
        """
        result = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "status": "failed",
            "corners": None,
            "fen": None,
            "error": None
        }
        
        try:
            # Get corners (auto-detect or use provided)
            if auto_corners is None:
                auto_corners = self._get_auto_corners(str(image_path))
            
            if auto_corners is None:
                result["error"] = "Auto-detection failed"
                return result
            
            # Display image with auto-detected corners
            image = cv2.imread(str(image_path))
            if image is None:
                result["error"] = "Could not load image"
                return result
            
            # Show auto-detected corners
            vis_img = self._create_visualization(str(image_path), auto_corners, "Auto-detected")
            cv2.imshow("Auto-detected Corners", cv2.resize(vis_img, (800, 600)))
            
            print(f"\n📸 Processing: {image_path.name}")
            print("=" * 50)
            print("Auto-detected corners shown in window.")
            print("Options:")
            print("  'y' - Accept auto-detected corners")
            print("  'm' - Manually adjust corners")
            print("  's' - Skip this image")
            print("  'q' - Quit processing")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):
                    # Accept auto-detected corners
                    corners = auto_corners
                    print("✅ Using auto-detected corners")
                    break
                    
                elif key == ord('m'):
                    # Manual adjustment (simplified)
                    print("Enter 4 corners as: x1,y1 x2,y2 x3,y3 x4,y4")
                    corners_input = input("Corners: ").strip()
                    
                    if corners_input.lower() == 'skip':
                        result["status"] = "skipped"
                        return result
                    
                    try:
                        corner_pairs = corners_input.split()
                        corners = []
                        for pair in corner_pairs:
                            x, y = map(float, pair.split(','))
                            corners.append([x, y])
                        
                        if len(corners) == 4:
                            print("✅ Using manual corners")
                            break
                        else:
                            print("❌ Must provide exactly 4 corners")
                            continue
                    except Exception as e:
                        print(f"❌ Invalid format: {e}")
                        continue
                        
                elif key == ord('s'):
                    result["status"] = "skipped"
                    return result
                    
                elif key == ord('q'):
                    result["status"] = "cancelled"
                    return result
            
            cv2.destroyAllWindows()
            
            # Get FEN position
            print("\n♟️  Enter FEN position:")
            print("Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            
            while True:
                fen_input = input("FEN: ").strip()
                
                if not fen_input:
                    print("❌ Empty FEN. Enter valid FEN or 'skip'")
                    continue
                
                if fen_input.lower() == 'skip':
                    result["status"] = "skipped"
                    return result
                
                # Add default ending if not provided
                if ' ' not in fen_input:
                    fen_input += " w KQkq - 0 1"
                
                if self._validate_fen(fen_input):
                    result["fen"] = fen_input
                    break
                else:
                    print("❌ Invalid FEN. Please check format.")
            
            # Save annotation
            annotation = {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "chess_set": self.chess_set,
                "corners": corners,
                "fen": result["fen"],
                "annotation_method": "semi_automated_batch",
                "corner_detection_api": "robust_port_8005",
                "timestamp": datetime.now().isoformat()
            }
            
            annotation_file = self.output_dir / "annotations" / f"{image_path.stem}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            # Create final visualization
            self._create_visualization(str(image_path), corners, "Final")
            
            result["corners"] = corners
            result["status"] = "completed"
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {image_path.name}: {e}")
        
        return result
    
    def batch_process(self, start_index: int = 0, max_images: Optional[int] = None):
        """
        Process images in batch
        """
        total_images = len(self.image_files)
        end_index = min(start_index + max_images, total_images) if max_images else total_images
        
        print(f"\n🚀 BATCH PROCESSING")
        print("=" * 60)
        print(f"📁 Total images: {total_images}")
        print(f"🎯 Processing: {start_index} to {end_index-1}")
        print(f"♟️  Chess set: {self.chess_set}")
        print("=" * 60)
        
        for i in range(start_index, end_index):
            image_path = self.image_files[i]
            
            # Skip if already processed
            if str(image_path) in self.progress["completed_files"]:
                print(f"⏭️  Skipping already processed: {image_path.name}")
                continue
            
            print(f"\n📊 Progress: {i + 1}/{total_images} ({((i + 1) / total_images) * 100:.1f}%)")
            
            # Process image
            result = self.process_single_image(image_path)
            
            # Update progress
            if result["status"] == "completed":
                self.progress["completed"] += 1
                self.progress["completed_files"].append(str(image_path))
                print(f"✅ Completed: {image_path.name}")
                
            elif result["status"] == "skipped":
                self.progress["skipped"] += 1
                self.progress["skipped_files"].append(str(image_path))
                print(f"⏭️  Skipped: {image_path.name}")
                
            elif result["status"] == "cancelled":
                print(f"🛑 Processing cancelled at {image_path.name}")
                break
                
            else:
                self.progress["failed"] += 1
                self.progress["failed_files"].append(str(image_path))
                print(f"❌ Failed: {image_path.name} - {result.get('error', 'Unknown error')}")
            
            self.progress["last_processed"] = str(image_path)
            self._save_progress()
        
        # Generate final report
        self._generate_report()
    
    def _generate_report(self):
        """Generate processing report"""
        report = {
            "processing_summary": {
                "total_images": len(self.image_files),
                "completed": self.progress["completed"],
                "skipped": self.progress["skipped"],
                "failed": self.progress["failed"],
                "completion_rate": (self.progress["completed"] / len(self.image_files)) * 100 if self.image_files else 0
            },
            "chess_set": self.chess_set,
            "start_time": self.progress["start_time"],
            "end_time": datetime.now().isoformat(),
            "completed_files": self.progress["completed_files"],
            "skipped_files": self.progress["skipped_files"],
            "failed_files": self.progress["failed_files"]
        }
        
        # Save report
        report_file = self.output_dir / "reports" / f"annotation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✅ Completed: {self.progress['completed']}")
        print(f"⏭️  Skipped: {self.progress['skipped']}")
        print(f"❌ Failed: {self.progress['failed']}")
        print(f"📊 Completion rate: {report['processing_summary']['completion_rate']:.1f}%")
        print(f"📁 Report saved: {report_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch Chess Annotation Helper")
    parser.add_argument("images_dir", help="Directory containing chess images")
    parser.add_argument("--output", "-o", default="./chess_set2_annotations", 
                       help="Output directory for annotations")
    parser.add_argument("--chess-set", "-s", default="set2", 
                       help="Chess set identifier")
    parser.add_argument("--start", type=int, default=0, 
                       help="Start index for processing")
    parser.add_argument("--max", type=int, 
                       help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    # Check if robust API is available
    try:
        response = requests.get("http://localhost:8005/health", timeout=5)
        if response.status_code != 200:
            print("❌ Robust corner detection API not available. Please start it first:")
            print("   python robust_corner_api.py")
            return
    except:
        print("❌ Robust corner detection API not available. Please start it first:")
        print("   python robust_corner_api.py")
        return
    
    # Create helper and process
    helper = BatchAnnotationHelper(args.images_dir, args.output, args.chess_set)
    helper.batch_process(args.start, args.max)

if __name__ == "__main__":
    main()
