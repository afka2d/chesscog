#!/usr/bin/env python3
"""
Quick validation test for Marshall models
Run this after training to verify performance
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 Running Quick Marshall Model Validation")
    logger.info("This will test the models against unseen data to check for overfitting")
    
    try:
        # Run the validation script
        result = subprocess.run([sys.executable, "validate_marshall_models.py"], 
                              capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            logger.info("✅ Validation completed successfully!")
            print("\n" + "="*60)
            print("VALIDATION RESULTS:")
            print("="*60)
            print(result.stdout)
        else:
            logger.error("❌ Validation failed!")
            print("Error:", result.stderr)
            
    except subprocess.TimeoutExpired:
        logger.warning("⏰ Validation timed out after 10 minutes")
    except Exception as e:
        logger.error(f"❌ Error running validation: {e}")

if __name__ == "__main__":
    main()
