#!/usr/bin/env python3
"""
Minimal Marshall API to test the loading issue.
"""

import logging
import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal Marshall API")

# Piece type labels
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}

def _get_piece_type_model_architecture(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
        if not model_path.exists():
            logger.error(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        # Create the model architecture first
        model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        logger.info("✅ Combined piece classifier architecture created")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("✅ Combined piece classifier weights loaded")
        
        model.eval()
        logger.info("✅ Combined piece classifier set to eval mode")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading combined piece classifier: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Minimal Marshall API...")
    
    # Test piece classifier loading
    logger.info("Loading combined piece type classifier...")
    piece_type_model = load_combined_piece_classifier()
    if piece_type_model is None:
        logger.error("❌ Failed to load combined piece classifier")
        raise RuntimeError("Combined piece classifier not found")
    logger.info("✅ Combined piece type classifier loaded successfully")
    
    logger.info("🎉 Minimal Marshall API startup completed successfully")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Minimal Marshall API is running"}

if __name__ == "__main__":
    print("🚀 Starting Minimal Marshall API on port 8004")
    uvicorn.run(app, host="0.0.0.0", port=8004)
