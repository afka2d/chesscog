#!/usr/bin/env python3
"""
Comprehensive API baseline test suite to ensure no performance degradation.
This captures the exact current behavior of your working API.
"""

import requests
import json
import logging
import cv2
import numpy as np
from pathlib import Path
import time
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIBaselineTest:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.baseline_results = {}
        self.test_images = []
        
    def capture_baseline(self):
        """Capture comprehensive baseline of current API performance"""
        print("🔍 CAPTURING API BASELINE")
        print("=" * 50)
        
        # Test API health
        health_result = self.test_api_health()
        if not health_result['success']:
            print("❌ API not healthy - cannot proceed")
            return False
        
        # Find test images
        self.discover_test_images()
        
        # Test each image with different corner methods
        for image_path in self.test_images:
            print(f"\n📸 Testing image: {Path(image_path).name}")
            self.test_image_comprehensive(image_path)
        
        # Save baseline
        self.save_baseline()
        
        # Generate baseline report
        self.generate_baseline_report()
        
        print("\n✅ BASELINE CAPTURED SUCCESSFULLY")
        return True
    
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data}")
                return {'success': True, 'data': health_data}
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
                return {'success': False, 'error': f"Status {response.status_code}"}
        except Exception as e:
            print(f"❌ API Health Check Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def discover_test_images(self):
        """Discover available test images"""
        dataset_path = "my_chess_images/train/images"
        if os.path.exists(dataset_path):
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            for ext in image_extensions:
                self.test_images.extend(Path(dataset_path).glob(f"**/*{ext}"))
        
        # Filter out JSON files that might have image extensions
        self.test_images = [img for img in self.test_images if self.is_valid_image(str(img))]
        
        print(f"📁 Found {len(self.test_images)} test images")
        for img in self.test_images:
            print(f"   - {img.name}")
    
    def is_valid_image(self, image_path):
        """Check if file is a valid image"""
        try:
            img = cv2.imread(image_path)
            return img is not None
        except:
            return False
    
    def test_image_comprehensive(self, image_path):
        """Test image with all corner detection methods"""
        image_name = Path(image_path).name
        self.baseline_results[image_name] = {}
        
        # Test with different corner methods
        corner_methods = [
            ("auto_detected", self.detect_corners_auto),
            ("working_from_logs", lambda: [[302.3999938964844, 302.3999938964844], [3729.60009765625, 302.3999938964844], [3729.60009765625, 2721.60009765625], [302.3999938964844, 2721.60009765625]]),
            ("estimated", self.detect_corners_estimated)
        ]
        
        for method_name, corner_func in corner_methods:
            print(f"  🔧 Testing {method_name} corners...")
            
            if method_name == "auto_detected":
                corners = corner_func(image_path)
            elif method_name == "estimated":
                corners = corner_func(image_path)
            else:
                corners = corner_func()
            
            if corners is None:
                print(f"    ❌ Could not get corners for {method_name}")
                continue
            
            # Test API call
            result = self.test_api_call(image_path, corners, method_name)
            if result:
                self.baseline_results[image_name][method_name] = result
                print(f"    ✅ {method_name}: {result['pieces_detected']} pieces, {result['occupied_squares']} occupied")
            else:
                print(f"    ❌ {method_name}: API call failed")
    
    def detect_corners_auto(self, image_path):
        """Auto-detect corners"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try different chessboard sizes
        for pattern_size in [(7, 7), (8, 8), (9, 9)]:
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners_2d = corners.reshape(-1, 2)
                
                # Get the 4 corner points
                top_left = corners_2d[np.argmin(corners_2d[:, 0] + corners_2d[:, 1])]
                top_right = corners_2d[np.argmax(corners_2d[:, 0] - corners_2d[:, 1])]
                bottom_right = corners_2d[np.argmax(corners_2d[:, 0] + corners_2d[:, 1])]
                bottom_left = corners_2d[np.argmin(corners_2d[:, 0] - corners_2d[:, 1])]
                
                return [top_left.tolist(), top_right.tolist(), bottom_right.tolist(), bottom_left.tolist()]
        
        return None
    
    def detect_corners_estimated(self, image_path):
        """Estimate corners"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        h, w = img.shape[:2]
        margin = min(h, w) * 0.1
        return [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ]
    
    def test_api_call(self, image_path, corners, method_name):
        """Test single API call and capture comprehensive results"""
        try:
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'corners': json.dumps(corners),
                    'debug': 'true'
                }
                
                response = requests.post(
                    f"{self.api_url}/recognize_chess_position_with_corners",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract comprehensive data
                pieces = result.get('pieces', [])
                occupancy = result.get('occupancy', [])
                fen = result.get('fen', '')
                debug_info = result.get('debug_info', {})
                success = result.get('success', False)
                
                # Calculate metrics
                pieces_detected = sum(1 for p in pieces if p is not None)
                occupied_squares = sum(occupancy)
                total_squares = len(occupancy)
                
                # Create comprehensive result
                comprehensive_result = {
                    'method': method_name,
                    'success': success,
                    'pieces_detected': pieces_detected,
                    'occupied_squares': occupied_squares,
                    'total_squares': total_squares,
                    'fen': fen,
                    'response_time': response_time,
                    'pieces_array': pieces,
                    'occupancy_array': occupancy,
                    'debug_info': debug_info,
                    'corners_used': corners,
                    'api_response_hash': hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()
                }
                
                return comprehensive_result
            else:
                return {
                    'method': method_name,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response_time': response_time
                }
                
        except Exception as e:
            return {
                'method': method_name,
                'success': False,
                'error': str(e),
                'response_time': 0
            }
    
    def save_baseline(self):
        """Save baseline results"""
        baseline_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_url': self.api_url,
            'test_images': [str(img) for img in self.test_images],
            'results': self.baseline_results
        }
        
        with open("api_baseline.json", "w") as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        print(f"\n💾 Baseline saved to: api_baseline.json")
    
    def generate_baseline_report(self):
        """Generate human-readable baseline report"""
        report = []
        report.append("# API Baseline Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"API URL: {self.api_url}")
        report.append(f"Images tested: {len(self.test_images)}")
        report.append("")
        
        # Summary statistics
        total_tests = 0
        successful_tests = 0
        total_pieces_detected = 0
        total_occupied_squares = 0
        
        for image_name, methods in self.baseline_results.items():
            report.append(f"## Image: {image_name}")
            report.append("-" * 30)
            
            for method_name, result in methods.items():
                total_tests += 1
                if result.get('success', False):
                    successful_tests += 1
                    total_pieces_detected += result.get('pieces_detected', 0)
                    total_occupied_squares += result.get('occupied_squares', 0)
                
                report.append(f"**{method_name}:**")
                report.append(f"  - Success: {result.get('success', False)}")
                report.append(f"  - Pieces: {result.get('pieces_detected', 0)}")
                report.append(f"  - Occupied: {result.get('occupied_squares', 0)}")
                report.append(f"  - FEN: {result.get('fen', 'N/A')}")
                report.append(f"  - Response time: {result.get('response_time', 0):.3f}s")
                report.append("")
        
        # Overall statistics
        report.append("## Overall Statistics")
        report.append("-" * 30)
        report.append(f"Success rate: {(successful_tests/total_tests)*100:.1f}% ({successful_tests}/{total_tests})")
        report.append(f"Average pieces per successful test: {total_pieces_detected/successful_tests:.1f}" if successful_tests > 0 else "No successful tests")
        report.append(f"Average occupied squares per successful test: {total_occupied_squares/successful_tests:.1f}" if successful_tests > 0 else "No successful tests")
        
        # Save report
        with open("api_baseline_report.md", "w") as f:
            f.write("\n".join(report))
        
        print(f"📊 Baseline report saved to: api_baseline_report.md")
    
    def compare_with_baseline(self, baseline_file="api_baseline.json"):
        """Compare current API performance with baseline"""
        if not os.path.exists(baseline_file):
            print(f"❌ Baseline file not found: {baseline_file}")
            return False
        
        print("🔍 COMPARING WITH BASELINE")
        print("=" * 50)
        
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Run current tests
        current_results = {}
        for image_path in self.test_images:
            image_name = Path(image_path).name
            if image_name in baseline_data['results']:
                print(f"\n📸 Testing image: {image_name}")
                self.test_image_comprehensive(image_path)
                current_results[image_name] = self.baseline_results[image_name]
        
        # Compare results
        differences = []
        for image_name in current_results:
            if image_name in baseline_data['results']:
                baseline_methods = baseline_data['results'][image_name]
                current_methods = current_results[image_name]
                
                for method_name in baseline_methods:
                    if method_name in current_methods:
                        baseline_result = baseline_methods[method_name]
                        current_result = current_methods[method_name]
                        
                        # Compare key metrics
                        if baseline_result.get('pieces_detected', 0) != current_result.get('pieces_detected', 0):
                            differences.append(f"{image_name}/{method_name}: Pieces changed from {baseline_result.get('pieces_detected', 0)} to {current_result.get('pieces_detected', 0)}")
                        
                        if baseline_result.get('occupied_squares', 0) != current_result.get('occupied_squares', 0):
                            differences.append(f"{image_name}/{method_name}: Occupied squares changed from {baseline_result.get('occupied_squares', 0)} to {current_result.get('occupied_squares', 0)}")
                        
                        if baseline_result.get('fen', '') != current_result.get('fen', ''):
                            differences.append(f"{image_name}/{method_name}: FEN changed")
        
        # Report results
        if differences:
            print("\n❌ DIFFERENCES DETECTED:")
            for diff in differences:
                print(f"  - {diff}")
            return False
        else:
            print("\n✅ NO DIFFERENCES DETECTED - API PERFORMANCE UNCHANGED")
            return True

def main():
    """Main function"""
    print("API Baseline Testing System")
    print("=" * 50)
    print("This will capture your API's current perfect performance")
    print("and create a baseline for safe cleanup operations.")
    print()
    
    tester = APIBaselineTest()
    
    action = input("Choose action:\n1. Capture baseline\n2. Compare with baseline\nEnter choice (1/2): ").strip()
    
    if action == "1":
        success = tester.capture_baseline()
        if success:
            print("\n🎯 BASELINE CAPTURED!")
            print("You can now safely make changes and use option 2 to verify no impact.")
    elif action == "2":
        success = tester.compare_with_baseline()
        if success:
            print("\n🎯 API PERFORMANCE VERIFIED!")
            print("Your changes have not impacted the API.")
        else:
            print("\n⚠️  PERFORMANCE CHANGES DETECTED!")
            print("Review the differences and consider rolling back.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
