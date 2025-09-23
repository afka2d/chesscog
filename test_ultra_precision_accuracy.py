#!/usr/bin/env python3
"""
Test Ultra Precision Corner Detection Accuracy
==============================================

Compare the new Ultra Precision API (port 8005) against existing methods
to validate the accuracy improvements within the 2-second budget.
"""

import requests
import json
import time
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraPrecisionAccuracyTester:
    def __init__(self):
        self.apis = {
            'yolo_only': 'http://localhost:8002',
            'fast_precision': 'http://localhost:8004', 
            'ultra_precision': 'http://localhost:8005'
        }
        self.test_images = [
            "my_chess_images/train/images/IMG_4698.JPG",
            "my_chess_images/train/images/IMG_4779.JPG",
            "my_chess_images/train/images/IMG_4763.JPG"
        ]
    
    def test_all_apis(self):
        """
        Test all corner detection APIs and compare performance
        """
        logger.info("🎯 ULTRA PRECISION ACCURACY TESTING")
        logger.info("=" * 60)
        logger.info("Testing new Ultra Precision API (port 8005) vs existing methods")
        logger.info("Target: <15px error in <2 seconds")
        logger.info("")
        
        # Check API availability
        available_apis = self._check_api_availability()
        
        if 'ultra_precision' not in available_apis:
            logger.error("❌ Ultra Precision API (port 8005) not available!")
            return
        
        results = {}
        
        for image_path in self.test_images:
            if not Path(image_path).exists():
                logger.warning(f"⚠️  Skipping missing image: {image_path}")
                continue
            
            logger.info(f"\n📸 Testing image: {Path(image_path).name}")
            image_results = {}
            
            # Test each available API
            for api_name, api_url in self.apis.items():
                if api_name not in available_apis:
                    logger.info(f"   {api_name}: ⚠️  Not available")
                    continue
                
                try:
                    # Test with 2-second budget for ultra precision
                    if api_name == 'ultra_precision':
                        result = self._test_ultra_precision_api(api_url, image_path, time_budget=2.0)
                    else:
                        result = self._test_standard_api(api_url, image_path)
                    
                    image_results[api_name] = result
                    
                    if result['success']:
                        logger.info(f"   {api_name}: ✅ {result['time_taken']:.3f}s")
                    else:
                        logger.info(f"   {api_name}: ❌ Failed")
                        
                except Exception as e:
                    logger.warning(f"   {api_name}: ❌ Error: {e}")
                    image_results[api_name] = {'success': False, 'error': str(e)}
            
            results[Path(image_path).name] = image_results
        
        # Generate comparison report
        self._generate_accuracy_comparison_report(results)
        
        return results
    
    def _check_api_availability(self):
        """
        Check which APIs are currently running
        """
        available = []
        
        for api_name, api_url in self.apis.items():
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    available.append(api_name)
                    logger.info(f"✅ {api_name} API available ({api_url})")
                else:
                    logger.warning(f"⚠️  {api_name} API unhealthy ({api_url})")
            except Exception as e:
                logger.warning(f"❌ {api_name} API not available ({api_url})")
        
        return available
    
    def _test_ultra_precision_api(self, api_url: str, image_path: str, time_budget: float = 2.0):
        """
        Test the Ultra Precision API with specific time budget
        """
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{api_url}/detect_corners", 
                files=files, 
                params={'time_budget': time_budget},
                timeout=time_budget + 5
            )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': data.get('processing_time', total_time),
                'budget_met': data.get('budget_met', True),
                'features_used': data.get('features_used', []),
                'accuracy_level': data.get('accuracy_level', 'unknown')
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}",
                'time_taken': total_time
            }
    
    def _test_standard_api(self, api_url: str, image_path: str):
        """
        Test standard corner detection APIs
        """
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect_corners", files=files, timeout=10)
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'corners': data.get('corners'),
                'time_taken': total_time,
                'budget_met': True,  # No budget concept for these APIs
                'features_used': [],
                'accuracy_level': 'standard'
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}",
                'time_taken': total_time
            }
    
    def _generate_accuracy_comparison_report(self, results):
        """
        Generate detailed comparison report
        """
        logger.info("\n" + "="*60)
        logger.info("🏆 ULTRA PRECISION ACCURACY COMPARISON REPORT")
        logger.info("="*60)
        
        # Calculate average performance
        api_performance = {}
        
        for api_name in self.apis.keys():
            times = []
            success_count = 0
            total_count = 0
            
            for image_name, image_results in results.items():
                if api_name in image_results:
                    result = image_results[api_name]
                    total_count += 1
                    
                    if result.get('success', False):
                        success_count += 1
                        times.append(result.get('time_taken', 0))
            
            if times:
                api_performance[api_name] = {
                    'avg_time': np.mean(times),
                    'success_rate': success_count / total_count * 100,
                    'total_tests': total_count
                }
        
        # Print performance table
        logger.info("\n📊 PERFORMANCE SUMMARY:")
        logger.info(f"{'API':<20} {'Avg Time':<12} {'Success Rate':<15} {'Tests':<8}")
        logger.info("-" * 60)
        
        for api_name, perf in api_performance.items():
            logger.info(f"{api_name:<20} {perf['avg_time']:.3f}s{'':<6} {perf['success_rate']:.1f}%{'':<9} {perf['total_tests']}")
        
        # Specific Ultra Precision analysis
        if 'ultra_precision' in api_performance:
            ultra_perf = api_performance['ultra_precision']
            logger.info(f"\n🎯 ULTRA PRECISION ANALYSIS:")
            logger.info(f"   Average Time: {ultra_perf['avg_time']:.3f}s (target: <2.0s)")
            logger.info(f"   Success Rate: {ultra_perf['success_rate']:.1f}% (target: 100%)")
            
            if ultra_perf['avg_time'] < 2.0:
                logger.info("   ✅ MEETS TIME BUDGET REQUIREMENT")
            else:
                logger.info("   ❌ EXCEEDS TIME BUDGET REQUIREMENT")
            
            # Compare with Fast Precision
            if 'fast_precision' in api_performance:
                fast_perf = api_performance['fast_precision']
                time_overhead = ultra_perf['avg_time'] - fast_perf['avg_time']
                logger.info(f"   Time overhead vs Fast Precision: +{time_overhead:.3f}s")
                
                if time_overhead < 0.5:
                    logger.info("   ✅ ACCEPTABLE OVERHEAD")
                else:
                    logger.info("   ⚠️  HIGH OVERHEAD")
        
        # Save detailed results
        self._save_detailed_results(results, api_performance)
        
        logger.info(f"\n💾 Detailed results saved to: ultra_precision_test_results.json")
        logger.info("🎯 Next step: Visual accuracy comparison with ground truth")
    
    def _save_detailed_results(self, results, api_performance):
        """
        Save detailed test results to file
        """
        output_data = {
            'test_timestamp': time.time(),
            'test_summary': {
                'target_accuracy': '<15px average error',
                'target_time': '<2 seconds',
                'apis_tested': list(self.apis.keys()),
                'images_tested': list(results.keys())
            },
            'api_performance': api_performance,
            'detailed_results': results
        }
        
        with open('ultra_precision_test_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)

def main():
    """
    Run the Ultra Precision accuracy testing
    """
    tester = UltraPrecisionAccuracyTester()
    results = tester.test_all_apis()
    
    if results:
        logger.info("\n🎉 Ultra Precision testing completed!")
        logger.info("📊 Check ultra_precision_test_results.json for detailed results")
    else:
        logger.error("❌ Testing failed - check API availability")

if __name__ == "__main__":
    main()
