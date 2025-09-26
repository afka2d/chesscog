#!/usr/bin/env python3
"""
Check Marshall model status and basic validation
"""

import os
import json
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_file(model_path, model_name):
    """Check if a model file exists and get basic info"""
    if not model_path.exists():
        logger.warning(f"❌ {model_name} not found: {model_path}")
        return False
    
    # Get file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ {model_name} found: {size_mb:.1f} MB")
    
    # Try to load the model to check if it's valid
    try:
        if model_name == "Occupancy":
            # For occupancy, it's a full model
            model = torch.load(str(model_path), map_location='cpu', weights_only=False)
            logger.info(f"   Model type: Full model with architecture")
        else:
            # For color/piece, it should be state_dict
            state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
            logger.info(f"   Model type: State dict with {len(state_dict)} parameters")
        
        logger.info(f"   ✅ Model loads successfully")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error loading model: {e}")
        return False

def check_annotations():
    """Check Marshall annotations status"""
    annotations_path = Path("marshall_chess_annotations/annotations.json")
    if not annotations_path.exists():
        logger.error("❌ Marshall annotations not found!")
        return False
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    completed_count = data.get('completed_count', 0)
    total_images = data.get('total_images', 0)
    
    logger.info(f"📊 Marshall Annotations Status:")
    logger.info(f"   Total annotations: {len(annotations)}")
    logger.info(f"   Excluded images: {len(excluded_images)}")
    logger.info(f"   Completed count: {completed_count}")
    logger.info(f"   Total images: {total_images}")
    
    # Calculate remaining
    remaining = total_images - len(annotations) - len(excluded_images)
    logger.info(f"   Remaining to annotate: {remaining}")
    
    return True

def main():
    """Check Marshall model status"""
    logger.info("🔍 Marshall Model Status Check")
    logger.info("=" * 50)
    
    # Check annotations
    check_annotations()
    
    logger.info(f"\n📁 Checking Marshall Models:")
    
    # Check each model
    models_to_check = [
        ("models_marshall_improved/occupancy_marshall.pt", "Occupancy Detection"),
        ("models_marshall_improved/color_classification_marshall.pt", "Color Classification"),
        ("models_marshall_improved/piece_classification_marshall.pt", "Piece Classification")
    ]
    
    available_models = []
    for model_path, model_name in models_to_check:
        path = Path(model_path)
        if check_model_file(path, model_name):
            available_models.append(model_name)
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Available models: {len(available_models)}")
    logger.info(f"   Models: {', '.join(available_models) if available_models else 'None'}")
    
    if available_models:
        logger.info(f"\n✅ Marshall training is progressing!")
        if len(available_models) == 3:
            logger.info("🎉 All Marshall models are ready!")
        else:
            logger.info(f"⏳ {3 - len(available_models)} models still training...")
    else:
        logger.warning("⚠️  No Marshall models found yet")
    
    # Check if training is still running
    logger.info(f"\n🔄 Training Status:")
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_occupancy_marshall.py' in result.stdout:
            logger.info("   ✅ Occupancy training is running")
        else:
            logger.info("   ⏸️  Occupancy training completed")
            
        if 'train_color_marshall.py' in result.stdout:
            logger.info("   ✅ Color training is running")
        else:
            logger.info("   ⏸️  Color training not running")
            
        if 'train_piece_marshall.py' in result.stdout:
            logger.info("   ✅ Piece training is running")
        else:
            logger.info("   ⏸️  Piece training not running")
            
    except Exception as e:
        logger.warning(f"   Could not check training status: {e}")

if __name__ == "__main__":
    main()
