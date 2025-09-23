#!/usr/bin/env python3
"""
Test Ultra Precision with Real Ground Truth
===========================================

Test the Ultra Precision API against real annotated corner data
from the grey background dataset to measure actual accuracy improvements.
"""

import requests
import json
import cv2
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraPrecisionGroundTruthTester:
    def __init__(self):
        self.apis = {
            'YOLO Only': 'http://localhost:8002',
            'Fast Precision': 'http://localhost:8004',
            'Ultra Precision': 'http://localhost:8005'
        }
        
        # Test cases with real annotations
        self.test_cases = [
            {
                'image': 'grey_background_dataset/images/val/IMG_4779.JPG',
                'annotation': 'grey_background_dataset/annotations/train/IMG_4779.json'
            },
            {
                'image': 'grey_background_dataset/images/test/IMG_4763.JPG', 
                'annotation': 'grey_background_dataset/annotations/train/IMG_4763.json'
            },
            {
                'image': 'grey_background_dataset/images/test/IMG_4785.JPG',
                'annotation': 'grey_background_dataset/annotations/train/IMG_4785.json'
            }
        ]
    
    def run_comprehensive_accuracy_test(self):
        """
        Run comprehensive accuracy test with real ground truth data
        """
        logger.info("🎯 ULTRA PRECISION GROUND TRUTH ACCURACY TEST")
        logger.info("=" * 60)
        logger.info("Testing against real annotated corner data")
        logger.info("Target: <15px average error in <2 seconds")
        logger.info("")
        
        # Check API availability
        available_apis = self._check_api_availability()
        
        if 'Ultra Precision' not in available_apis:
            logger.error("❌ Ultra Precision API (port 8005) not available!")
            return
        
        all_results = []
        
        for test_case in self.test_cases:
            image_path = test_case['image']
            annotation_path = test_case['annotation']
            
            if not Path(image_path).exists() or not Path(annotation_path).exists():
                logger.warning(f"⚠️  Skipping missing files: {Path(image_path).name}")
                continue
            
            logger.info(f"\n📸 Testing: {Path(image_path).name}")
            
            # Load ground truth
            ground_truth = self._load_ground_truth(annotation_path)
            if ground_truth is None:
                continue
            
            # Test each API
            test_results = {'image': Path(image_path).name, 'ground_truth': ground_truth}
            
            for api_name in available_apis:
                api_url = self.apis[api_name]
                
                try:
                    if 'Ultra Precision' in api_name:
                        result = self._test_ultra_precision_api(api_url, image_path, time_budget=2.0)
                    else:
                        result = self._test_standard_api(api_url, image_path)
                    
                    if result['success']:
                        # Calculate accuracy
                        error = self._calculate_corner_error(ground_truth, result['corners'])
                        result['accuracy_error'] = error
                        
                        # Performance rating
                        if error < 15:
                            rating = "🏆 EXCELLENT"
                        elif error < 25:
                            rating = "✅ GOOD"  
                        elif error < 50:
                            rating = "⚠️  FAIR"
                        else:
                            rating = "❌ POOR"
                        
                        logger.info(f"   {api_name}: {error:.1f}px {rating} ({result['time_taken']:.3f}s)")
                        
                    else:
                        logger.warning(f"   {api_name}: ❌ Failed")
                        result['accuracy_error'] = float('inf')
                    
                    test_results[api_name] = result
                    
                except Exception as e:
                    logger.warning(f"   {api_name}: ❌ Error: {e}")
                    test_results[api_name] = {'success': False, 'error': str(e), 'accuracy_error': float('inf')}
            
            all_results.append(test_results)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _check_api_availability(self):
        """
        Check which APIs are running
        """
        available = []
        
        for api_name, api_url in self.apis.items():
            try:
                response = requests.get(f"{api_url}/health", timeout=3)
                if response.status_code == 200:
                    available.append(api_name)
            except:
                pass
        
        logger.info(f"Available APIs: {', '.join(available)}")
        return available
    
    def _load_ground_truth(self, annotation_path: str):
        """
        Load ground truth corners
        """
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            return data.get('corners')
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return None
    
    def _test_ultra_precision_api(self, api_url: str, image_path: str, time_budget: float):
        """
        Test Ultra Precision API
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{api_url}/detect_corners",
                files=files,
                params={'time_budget': time_budget},
                timeout=time_budget + 5
            )
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': data.get('processing_time'),
                'budget_met': data.get('budget_met'),
                'features_used': data.get('features_used', [])
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _test_standard_api(self, api_url: str, image_path: str):
        """
        Test standard APIs
        """
        import time
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect_corners", files=files, timeout=10)
        
        time_taken = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': time_taken
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _calculate_corner_error(self, ground_truth, predicted):
        """
        Calculate average pixel error between ground truth and predicted corners
        """
        if not ground_truth or not predicted:
            return float('inf')
        
        gt_np = np.array(ground_truth)
        pred_np = np.array(predicted)
        
        if gt_np.shape != pred_np.shape:
            return float('inf')
        
        # Calculate Euclidean distance for each corner
        errors = np.linalg.norm(gt_np - pred_np, axis=1)
        return np.mean(errors)
    
    def _generate_comprehensive_report(self, all_results):
        """
        Generate comprehensive accuracy report
        """
        logger.info("\n" + "="*70)
        logger.info("🏆 ULTRA PRECISION COMPREHENSIVE ACCURACY REPORT")
        logger.info("="*70)
        
        # Calculate overall statistics
        api_stats = {}
        
        for api_name in self.apis.keys():
            errors = []
            times = []
            success_count = 0
            
            for result in all_results:
                if api_name in result and result[api_name].get('success', False):
                    success_count += 1
                    error = result[api_name].get('accuracy_error', float('inf'))
                    if error != float('inf'):
                        errors.append(error)
                        times.append(result[api_name].get('time_taken', 0))
            
            if errors:
                api_stats[api_name] = {
                    'avg_error': np.mean(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'avg_time': np.mean(times),
                    'success_rate': success_count / len(all_results) * 100
                }
        
        # Print statistics table
        logger.info("\n📊 ACCURACY STATISTICS:")
        logger.info(f"{'API':<20} {'Avg Error':<12} {'Min Error':<12} {'Max Error':<12} {'Avg Time':<10} {'Success'}")
        logger.info("-" * 85)
        
        for api_name, stats in api_stats.items():
            logger.info(f"{api_name:<20} {stats['avg_error']:.1f}px{'':<6} {stats['min_error']:.1f}px{'':<6} "
                       f"{stats['max_error']:.1f}px{'':<6} {stats['avg_time']:.3f}s{'':<4} {stats['success_rate']:.0f}%")
        
        # Ultra Precision specific analysis
        if 'Ultra Precision' in api_stats:
            ultra_stats = api_stats['Ultra Precision']
            logger.info(f"\n🎯 ULTRA PRECISION DETAILED ANALYSIS:")
            logger.info(f"   Target Error: <15px")
            logger.info(f"   Actual Error: {ultra_stats['avg_error']:.1f}px")
            
            if ultra_stats['avg_error'] < 15:
                logger.info("   ✅ MEETS ACCURACY TARGET")
            else:
                improvement_needed = ultra_stats['avg_error'] - 15
                logger.info(f"   ⚠️  Needs {improvement_needed:.1f}px improvement to meet target")
            
            logger.info(f"   Target Time: <2.0s")
            logger.info(f"   Actual Time: {ultra_stats['avg_time']:.3f}s")
            
            if ultra_stats['avg_time'] < 2.0:
                logger.info("   ✅ MEETS TIME TARGET")
            else:
                logger.info("   ❌ EXCEEDS TIME TARGET")
            
            # Compare with other methods
            if 'Fast Precision' in api_stats:
                fast_stats = api_stats['Fast Precision']
                error_improvement = fast_stats['avg_error'] - ultra_stats['avg_error']
                time_overhead = ultra_stats['avg_time'] - fast_stats['avg_time']
                
                logger.info(f"\n📈 IMPROVEMENT VS FAST PRECISION:")
                logger.info(f"   Error Improvement: {error_improvement:.1f}px ({error_improvement/fast_stats['avg_error']*100:.1f}%)")
                logger.info(f"   Time Overhead: +{time_overhead:.3f}s")
                
                if error_improvement > 5:
                    logger.info("   🏆 SIGNIFICANT ACCURACY IMPROVEMENT")
                elif error_improvement > 2:
                    logger.info("   ✅ MODERATE ACCURACY IMPROVEMENT")
                else:
                    logger.info("   ⚠️  MINIMAL ACCURACY IMPROVEMENT")
        
        # Save results
        self._save_results(all_results, api_stats)
        
        logger.info(f"\n💾 Results saved to: ultra_precision_ground_truth_results.json")
        logger.info("🎯 Ultra Precision testing completed!")
    
    def _save_results(self, all_results, api_stats):
        """
        Save comprehensive results
        """
        output_data = {
            'test_summary': {
                'target_accuracy': '<15px average error',
                'target_time': '<2 seconds',
                'test_description': 'Ultra Precision API vs existing methods with real ground truth',
                'images_tested': len(all_results)
            },
            'api_statistics': api_stats,
            'detailed_results': all_results
        }
        
        with open('ultra_precision_ground_truth_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)

def main():
    """
    Run the ground truth accuracy test
    """
    tester = UltraPrecisionGroundTruthTester()
    results = tester.run_comprehensive_accuracy_test()

if __name__ == "__main__":
    main()
