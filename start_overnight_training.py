#!/usr/bin/env python3
"""
Start overnight Marshall training after occupancy model completes
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🌙 Starting Overnight Marshall Training")
    logger.info("This will run color and piece classification after occupancy completes")
    
    # Run the overnight training script
    try:
        result = subprocess.run([sys.executable, "overnight_marshall_training.py"])
        if result.returncode == 0:
            logger.info("✅ Overnight training completed successfully!")
        else:
            logger.error("❌ Overnight training failed!")
    except Exception as e:
        logger.error(f"❌ Error running overnight training: {e}")

if __name__ == "__main__":
    main()
