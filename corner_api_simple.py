#!/usr/bin/env python3
"""
Simple corner detection API that runs on port 8002.
Completely separate from your main API on port 8001.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from pathlib import Path

app = FastAPI(title="Chess Corner Detection API", version="1.0.0")

class LightweightCornerModel(nn.Module):
    def __init__(self):
        super(LightweightCornerModel, self).__init__()
        
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        
        self.corner_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

# Global variables
corner_model = None
corner_transform = None
device = torch.device('cpu')

@app.on_event("startup")
async def startup_event():
    """Load corner detection model"""
    global corner_model, corner_transform
    
    print("🔧 Starting Corner Detection API...")
    
    try:
        model_path = "models/corner_detector_best.pt"
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            corner_model = LightweightCornerModel()
            corner_model.load_state_dict(checkpoint['model_state_dict'])
            corner_model.eval()
            
            corner_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ Corner detection model loaded successfully")
        else:
            print(f"⚠️  Model not found: {model_path}")
            corner_model = None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        corner_model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chess Corner Detection API",
        "version": "1.0.0",
        "model_loaded": corner_model is not None,
        "endpoints": [
            "POST /detect_corners - Detect corners and return coordinates",
            "POST /visualize_corners - Detect corners and return image with visualization",
            "GET /health - Health check"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": corner_model is not None,
        "port": 8002
    }

@app.post("/detect_corners")
async def detect_corners(image: UploadFile = File(...)):
    """Detect chess board corners"""
    if corner_model is None:
        raise HTTPException(status_code=503, detail="Corner detection model not loaded")
    
    try:
        # Read image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = cv_image_rgb.shape[:2]
        
        # Prepare image
        pil_image = Image.fromarray(cv_image_rgb)
        input_tensor = corner_transform(pil_image).unsqueeze(0)
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = corner_model(input_tensor).numpy()[0]
        
        # Convert to pixel coordinates
        corners_pixels = corners_normalized.reshape(4, 2)
        corners_pixels[:, 0] *= orig_w
        corners_pixels[:, 1] *= orig_h
        
        return {
            "success": True,
            "corners": corners_pixels.tolist(),
            "image_dimensions": [orig_w, orig_h],
            "model": "lightweight_corner_detector"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner detection failed: {str(e)}")

@app.post("/visualize_corners")
async def visualize_corners(image: UploadFile = File(...)):
    """Detect corners and return visualization"""
    if corner_model is None:
        raise HTTPException(status_code=503, detail="Corner detection model not loaded")
    
    try:
        # Read image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = cv_image_rgb.shape[:2]
        
        # Prepare image
        pil_image = Image.fromarray(cv_image_rgb)
        input_tensor = corner_transform(pil_image).unsqueeze(0)
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = corner_model(input_tensor).numpy()[0]
        
        # Convert to pixel coordinates
        corners_pixels = corners_normalized.reshape(4, 2)
        corners_pixels[:, 0] *= orig_w
        corners_pixels[:, 1] *= orig_h
        
        # Create visualization
        vis_image = cv_image.copy()
        
        # Draw corners
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_pixels, corner_colors, corner_labels)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_image, (x, y), 30, color, -1)
            cv2.putText(vis_image, label, (x-20, y-35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Draw board outline
        corners_int = corners_pixels.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 5)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, vis_image)
            temp_path = tmp_file.name
        
        return FileResponse(
            temp_path,
            media_type="image/jpeg",
            filename="corner_detection_result.jpg",
            background=None  # Don't delete the file immediately
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("Simple Corner Detection API")
    print("=" * 50)
    print("🛡️  SAFETY: Runs on port 8002 (separate from your main API)")
    print("   Your main API on port 8001 will NOT be affected")
    print()
    
    # Check if model exists
    model_path = Path("models/corner_detector_best.pt")
    if not model_path.exists():
        print("❌ Corner detection model not found!")
        print("   The model should have been created by the training script.")
        print("   Check if models/corner_detector_best.pt exists.")
        exit(1)
    
    print("🚀 Starting Corner Detection API on port 8002...")
    print("   Health check: http://localhost:8002/health")
    print("   API docs: http://localhost:8002/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
