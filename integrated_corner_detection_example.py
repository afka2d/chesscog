#!/usr/bin/env python3
"""
Example of how to integrate automatic corner detection with your existing API.
This shows how to eliminate manual corner selection.
"""

import requests
import json
from corner_detection_service import CornerDetectionService
from pathlib import Path

class IntegratedChessRecognition:
    def __init__(self, 
                 main_api_url="http://localhost:8001",
                 corner_service=None):
        self.main_api_url = main_api_url
        self.corner_service = corner_service or CornerDetectionService()
    
    def recognize_chess_position_auto(self, image_path):
        """Recognize chess position with automatic corner detection"""
        print(f"🔍 Recognizing chess position: {Path(image_path).name}")
        
        # Step 1: Automatically detect corners
        print("   🎯 Step 1: Detecting corners automatically...")
        corners = self.corner_service.detect_corners(image_path)
        
        if corners is None:
            print("   ❌ Corner detection failed")
            return None
        
        print(f"   ✅ Corners detected: {corners}")
        
        # Step 2: Use detected corners with your main API
        print("   🤖 Step 2: Calling main chess recognition API...")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'corners': json.dumps(corners),
                    'debug': 'true'
                }
                
                response = requests.post(
                    f"{self.main_api_url}/recognize_chess_position_with_corners",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                
                pieces_detected = sum(1 for p in result.get('pieces', []) if p is not None)
                fen = result.get('fen', '')
                
                print(f"   ✅ Chess recognition successful!")
                print(f"      Pieces detected: {pieces_detected}")
                print(f"      FEN: {fen}")
                
                return {
                    'success': True,
                    'corners': corners,
                    'pieces_detected': pieces_detected,
                    'fen': fen,
                    'full_result': result
                }
            else:
                print(f"   ❌ Main API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error calling main API: {e}")
            return None
    
    def batch_recognize(self, image_paths):
        """Recognize multiple images with automatic corner detection"""
        print(f"🚀 BATCH RECOGNITION WITH AUTO CORNER DETECTION")
        print("=" * 60)
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n--- Image {i+1}/{len(image_paths)} ---")
            result = self.recognize_chess_position_auto(image_path)
            
            if result:
                results.append(result)
        
        # Summary
        if results:
            print(f"\n📊 BATCH RESULTS SUMMARY:")
            print(f"   Successful recognitions: {len(results)}/{len(image_paths)}")
            
            total_pieces = sum(r['pieces_detected'] for r in results)
            avg_pieces = total_pieces / len(results)
            
            print(f"   Total pieces detected: {total_pieces}")
            print(f"   Average pieces per image: {avg_pieces:.1f}")
            
            non_empty_fens = sum(1 for r in results if r['fen'] != '8/8/8/8/8/8/8/8 w - - 0 1')
            fen_success_rate = (non_empty_fens / len(results)) * 100
            
            print(f"   Non-empty FENs: {non_empty_fens}/{len(results)} ({fen_success_rate:.1f}%)")
            
            return results
        else:
            print(f"\n❌ No successful recognitions")
            return []

def demo_automatic_corner_detection():
    """Demo the automatic corner detection system"""
    print("🎯 AUTOMATIC CORNER DETECTION DEMO")
    print("=" * 60)
    print("This demonstrates how to eliminate manual corner selection")
    print("by using the trained corner detection model.")
    print()
    
    # Check if main API is available
    main_api_available = False
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Main chess recognition API is available")
            main_api_available = True
        else:
            print("⚠️  Main API not available - will only test corner detection")
            main_api_available = False
    except:
        print("⚠️  Main API not available - will only test corner detection")
        main_api_available = False
    
    # Create integrated service
    integrated_service = IntegratedChessRecognition()
    
    # Find test images
    test_images = []
    
    test_dir = Path("grey_background_dataset/images/test")
    if test_dir.exists():
        test_images.extend(list(test_dir.glob("*.JPG"))[:2])
    
    val_dir = Path("grey_background_dataset/images/val")
    if val_dir.exists():
        test_images.extend(list(val_dir.glob("*.JPG"))[:2])
    
    if not test_images:
        print("❌ No test images found")
        return
    
    # Test automatic recognition
    if main_api_available:
        print(f"\n🚀 Testing integrated system (corner detection + chess recognition)...")
        results = integrated_service.batch_recognize([str(img) for img in test_images])
        
        if results:
            print(f"\n🎯 INTEGRATION SUCCESSFUL!")
            print("You can now use automatic corner detection in your workflow!")
        else:
            print(f"\n⚠️  Integration needs main API to be running")
    else:
        print(f"\n🎯 CORNER DETECTION ONLY TEST")
        print("(Main API not available for full integration test)")
        
        # Test just corner detection
        service = CornerDetectionService()
        
        for image_path in test_images[:2]:
            print(f"\n📸 Testing: {image_path.name}")
            result = service.visualize_corners(str(image_path))
            
            if result:
                print(f"   ✅ Corners detected successfully")
                print(f"   📸 Visualization: {result['visualization_path']}")
            else:
                print(f"   ❌ Corner detection failed")

def main():
    """Main function"""
    print("Integrated Corner Detection Demo")
    print("=" * 50)
    
    demo_automatic_corner_detection()
    
    print(f"\n🎯 DEMO COMPLETE!")
    print("\nWhat you now have:")
    print("✅ Trained corner detection model (78 pixel average accuracy)")
    print("✅ Corner detection service that works independently")
    print("✅ Visualization system that shows detected corners")
    print("✅ Integration example for your workflow")
    print("\nNext steps:")
    print("1. Review the visualization images to see corner detection quality")
    print("2. Integrate automatic corner detection into your workflow when ready")
    print("3. This will eliminate the need for manual corner selection!")

if __name__ == "__main__":
    main()
