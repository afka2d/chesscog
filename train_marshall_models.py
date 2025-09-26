#!/usr/bin/env python3
"""
Train Marshall Models Script
Runs the complete training pipeline for all Marshall improved models
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from marshall_training_pipeline import MarshallTrainingPipeline

def main():
    """Main training function"""
    print("🚀 Starting Marshall Model Training")
    print("=" * 60)
    print("This will create improved models using Marshall data")
    print("Your current working models will NOT be affected")
    print("=" * 60)
    
    # Check if Marshall annotations exist
    marshall_annotations = Path("marshall_chess_annotations/annotations.json")
    if not marshall_annotations.exists():
        print("❌ Marshall annotations not found!")
        print("Please run the annotation tool first to create training data")
        return
    
    # Check if Marshall photos exist
    marshall_photos = Path("/Users/tonyblum/Desktop/marshall photos")
    if not marshall_photos.exists():
        print("❌ Marshall photos directory not found!")
        print("Please ensure photos are in /Users/tonyblum/Desktop/marshall photos")
        return
    
    try:
        # Initialize training pipeline
        pipeline = MarshallTrainingPipeline()
        
        # Run training
        pipeline.run_training_pipeline()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Models saved to: models_marshall_improved/")
        print("🔗 You can now use the Marshall Improved API on port 8006")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

