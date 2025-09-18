#!/usr/bin/env python3
"""
Separate corner detection API endpoint.
This runs on a different port and won't affect your main working API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chess Corner Detection API", version="1.0.0")

# Global variables for model
corner_model = None
corner_transform = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CornerDetectionModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0'):
        super(CornerDetectionModel, self).__init__()
        
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.corner_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

@app.on_event("startup")
async def startup_event():
    """Load the corner detection model on startup"""
    global corner_model, corner_transform
    
    logger.info("🔧 Starting Corner Detection API...")
    
    try:
        # Load corner detection model
        model_path = "models/corner_detector_best.pt"
        
        if not Path(model_path).exists():
            logger.warning(f"⚠️  Corner detection model not found at {model_path}")
            logger.info("   Run train_corner_detection_model.py first to create the model")
            corner_model = None
            return
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        corner_model = CornerDetectionModel()
        corner_model.load_state_dict(checkpoint['model_state_dict'])
        corner_model = corner_model.to(device)
        corner_model.eval()
        
        # Create transform
        corner_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ Corner detection model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load corner detection model: {e}")
        corner_model = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chess Corner Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect_corners": "POST - Detect chess board corners in an image",
            "/visualize_corners": "POST - Detect corners and return visualization",
            "/health": "GET - Health check"
        },
        "model_loaded": corner_model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy - Corner Detection API",
        "model_loaded": corner_model is not None,
        "device": str(device),
        "port": 8002
    }

@app.post("/detect_corners")
async def detect_corners(image: UploadFile = File(...)):
    """Detect chess board corners in an image"""
    if corner_model is None:
        raise HTTPException(status_code=503, detail="Corner detection model not loaded")
    
    try:
        # Read image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_h, orig_w = cv_image.shape[:2]
        
        # Prepare image for model
        pil_image = Image.fromarray(cv_image)
        input_tensor = corner_transform(pil_image).unsqueeze(0).to(device)
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = corner_model(input_tensor).cpu().numpy()[0]
        
        # Convert back to original image coordinates
        corners_pixels = corners_normalized.reshape(4, 2)
        corners_pixels[:, 0] *= orig_w  # Scale x coordinates
        corners_pixels[:, 1] *= orig_h  # Scale y coordinates
        
        # Convert to list format
        corners_list = corners_pixels.tolist()
        
        return {
            "success": True,
            "corners": corners_list,
            "image_dimensions": [orig_w, orig_h],
            "confidence": "high" if corner_model else "low"
        }
        
    except Exception as e:
        logger.error(f"Error in corner detection: {e}")
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
        
        # Get original dimensions
        orig_h, orig_w = cv_image_rgb.shape[:2]
        
        # Prepare image for model
        pil_image = Image.fromarray(cv_image_rgb)
        input_tensor = corner_transform(pil_image).unsqueeze(0).to(device)
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = corner_model(input_tensor).cpu().numpy()[0]
        
        # Convert back to original image coordinates
        corners_pixels = corners_normalized.reshape(4, 2)
        corners_pixels[:, 0] *= orig_w
        corners_pixels[:, 1] *= orig_h
        
        # Draw corners on image
        vis_image = cv_image.copy()
        
        # Draw corners
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_pixels, corner_colors, corner_labels)):
            x, y = int(corner[0]), int(corner[1])
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 20, color, -1)
            
            # Draw label
            cv2.putText(vis_image, label, (x-15, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw board outline
        corners_int = corners_pixels.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 3)
        
        # Convert to base64 for return
        _, buffer = cv2.imencode('.jpg', vis_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "corners": corners_pixels.tolist(),
            "image_dimensions": [orig_w, orig_h],
            "visualization": img_base64,
            "visualization_format": "base64_jpg"
        }
        
    except Exception as e:
        logger.error(f"Error in corner visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Corner visualization failed: {str(e)}")

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Demo page for testing corner detection"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chess Corner Detection Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Chess Corner Detection Demo</h1>
            <p>Upload a chess board image to automatically detect corners.</p>
            
            <div class="upload-area">
                <input type="file" id="imageInput" accept="image/*" />
                <button onclick="detectCorners()">Detect Corners</button>
            </div>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            async function detectCorners() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                try {
                    resultDiv.innerHTML = '<p>🔍 Detecting corners...</p>';
                    
                    const response = await fetch('/visualize_corners', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        resultDiv.innerHTML = `
                            <h3>✅ Corners Detected!</h3>
                            <img src="data:image/jpeg;base64,${result.visualization}" alt="Corner Detection Result" />
                            <h4>Detected Corners:</h4>
                            <pre>${JSON.stringify(result.corners, null, 2)}</pre>
                        `;
                    } else {
                        resultDiv.innerHTML = '<p>❌ Corner detection failed</p>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    print("Chess Corner Detection API")
    print("=" * 50)
    print("⚠️  SAFETY: This runs on port 8002 (separate from your main API)")
    print("   Your main API on port 8001 will NOT be affected")
    print()
    
    # Check if model exists
    model_path = Path("models/corner_detector_best.pt")
    if not model_path.exists():
        print("❌ Corner detection model not found!")
        print("   Please run: python train_corner_detection_model.py")
        print("   This will train the corner detection model first.")
        exit(1)
    
    print("🚀 Starting Corner Detection API on port 8002...")
    print("   Demo page: http://localhost:8002/demo")
    print("   API docs: http://localhost:8002/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
