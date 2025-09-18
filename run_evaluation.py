#!/usr/bin/env python3
"""
Main evaluation script that runs comprehensive model accuracy tests.
"""

import os
import sys
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api_running():
    """Check if the local development API is running"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Local development API is running")
            return True
        else:
            logger.error("❌ Local development API is not responding correctly")
            return False
    except:
        logger.error("❌ Cannot connect to local development API")
        return False

def run_simple_evaluation():
    """Run simple evaluation without ground truth"""
    logger.info("Running simple evaluation (no ground truth required)...")
    
    try:
        from evaluate_simple import SimpleEvaluator
        
        evaluator = SimpleEvaluator()
        dataset_path = "my_chess_images/train/images"
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        evaluator.evaluate_dataset(dataset_path)
        return True
        
    except Exception as e:
        logger.error(f"Error running simple evaluation: {e}")
        return False

def run_detailed_evaluation():
    """Run detailed evaluation with ground truth"""
    logger.info("Running detailed evaluation (requires ground truth)...")
    
    try:
        from evaluate_model_accuracy import ModelEvaluator
        
        evaluator = ModelEvaluator()
        dataset_path = "my_chess_images/train/images"
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        evaluator.evaluate_dataset(dataset_path)
        return True
        
    except Exception as e:
        logger.error(f"Error running detailed evaluation: {e}")
        return False

def create_ground_truth():
    """Create ground truth annotations"""
    logger.info("Creating ground truth annotations...")
    
    try:
        from create_ground_truth import GroundTruthCreator
        
        creator = GroundTruthCreator()
        dataset_path = "my_chess_images/train/images"
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        creator.create_annotations_for_dataset(dataset_path, max_images=3)
        return True
        
    except Exception as e:
        logger.error(f"Error creating ground truth: {e}")
        return False

def main():
    """Main evaluation function"""
    print("Chess Model Accuracy Evaluation")
    print("=" * 50)
    
    # Check if API is running
    if not check_api_running():
        print("\nPlease start the local development API first:")
        print("  ./start_local_dev.sh")
        return
    
    print("\nEvaluation Options:")
    print("1. Simple evaluation (no ground truth required)")
    print("2. Detailed evaluation (requires ground truth annotations)")
    print("3. Create ground truth annotations first")
    print("4. Run all evaluations")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        logger.info("Running simple evaluation...")
        run_simple_evaluation()
        
    elif choice == "2":
        logger.info("Running detailed evaluation...")
        run_detailed_evaluation()
        
    elif choice == "3":
        logger.info("Creating ground truth annotations...")
        create_ground_truth()
        
    elif choice == "4":
        logger.info("Running all evaluations...")
        
        # First create some ground truth
        print("\nStep 1: Creating ground truth annotations...")
        if create_ground_truth():
            print("✅ Ground truth created successfully")
        else:
            print("❌ Failed to create ground truth")
        
        # Then run both evaluations
        print("\nStep 2: Running simple evaluation...")
        if run_simple_evaluation():
            print("✅ Simple evaluation completed")
        else:
            print("❌ Simple evaluation failed")
        
        print("\nStep 3: Running detailed evaluation...")
        if run_detailed_evaluation():
            print("✅ Detailed evaluation completed")
        else:
            print("❌ Detailed evaluation failed")
        
        print("\n🎉 All evaluations completed!")
        
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
