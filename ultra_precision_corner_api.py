#!/usr/bin/env python3
"""
Ultra Precision Corner Detection API
====================================

Maximum accuracy corner detection API running on port 8005.
Designed specifically for chess board warping accuracy requirements.

🎯 Target: <15px average error in <2 seconds
🛡️ Safety: Completely separate from all other APIs
⚡ Features: Multi-resolution YOLO, adaptive refinement, intelligent optimization

Endpoints:
- POST /detect_corners?time_budget=2.0
- POST /visualize_corners?time_budget=2.0  
- POST /compare_accuracy
- GET /health
- GET /demo
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import json
import time
import tempfile
import os
import base64
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our ultra precision detector
from ultra_precision_corner_detector import UltraPrecisionCornerDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ultra Precision Corner Detection API",
    description="Maximum accuracy corner detection for chess board warping. Port 8005.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
ultra_detector = None

@app.on_event("startup")
async def startup_event():
    global ultra_detector
    
    print("🚀 Starting Ultra Precision Corner Detection API...")
    print("🛡️  SAFETY: This runs on port 8005, separate from all other APIs")
    print("🎯 Maximum accuracy corner detection for precise board warping")
    print("⚡ Multi-resolution YOLO + adaptive refinement + intelligent optimization")
    print("📍 Endpoints:")
    print("   Health: http://localhost:8005/health")
    print("   Docs: http://localhost:8005/docs")
    print("   Demo: http://localhost:8005/demo")
    print("============================================================")
    
    logger.info("🔧 Loading ultra precision detection system...")
    
    try:
        ultra_detector = UltraPrecisionCornerDetector()
        logger.info("✅ Ultra precision detection system loaded successfully")
        print("✅ Ultra precision detection system loaded successfully")
        print("🎯 Target: <15px error in <2 seconds")
        print("🏆 Multi-stage pipeline: Multi-YOLO → Adaptive Sub-pixel → Smart Geometric → Selective Edge")
        
    except Exception as e:
        logger.error(f"❌ Failed to load ultra precision system: {e}")
        print(f"❌ Failed to load ultra precision system: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "Ultra Precision Corner Detection API",
        "port": 8005,
        "detector_loaded": ultra_detector is not None,
        "target_accuracy": "<15px average error",
        "target_speed": "<2 seconds",
        "features": [
            "Multi-resolution YOLO ensemble",
            "Adaptive sub-pixel refinement", 
            "Intelligent geometric optimization",
            "Selective edge enhancement"
        ]
    })

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head><title>Ultra Precision Corner Detection API</title></head>
        <body>
            <h1>🎯 Ultra Precision Corner Detection API</h1>
            <p><strong>Port:</strong> 8005</p>
            <p><strong>Target:</strong> &lt;15px error in &lt;2 seconds</p>
            <p><strong>Status:</strong> Maximum accuracy corner detection</p>
            
            <h2>🚀 Key Features:</h2>
            <ul>
                <li>Multi-resolution YOLO ensemble (640px + 896px)</li>
                <li>Adaptive sub-pixel refinement based on image quality</li>
                <li>Intelligent geometric optimization with confidence weighting</li>
                <li>Selective edge enhancement for uncertain corners</li>
                <li>Time budget management with graceful degradation</li>
            </ul>
            
            <h2>📍 Endpoints:</h2>
            <ul>
                <li><a href="/docs">📚 API Documentation</a></li>
                <li><a href="/demo">🎮 Interactive Demo</a></li>
                <li><a href="/health">💊 Health Check</a></li>
            </ul>
            
            <h2>🎯 Usage:</h2>
            <pre>
# Basic detection (2s budget)
curl -X POST -F "file=@image.jpg" http://localhost:8005/detect_corners

# Custom time budget
curl -X POST -F "file=@image.jpg" "http://localhost:8005/detect_corners?time_budget=1.5"

# Visualization
curl -X POST -F "file=@image.jpg" http://localhost:8005/visualize_corners
            </pre>
        </body>
    </html>
    """

@app.post("/detect_corners")
async def detect_corners_endpoint(file: UploadFile = File(...), time_budget: float = Query(2.0, ge=0.5, le=10.0)):
    """
    Detect corners with ultra precision
    
    Args:
        file: Image file
        time_budget: Maximum processing time in seconds (0.5-10.0)
    """
    if not ultra_detector:
        raise HTTPException(status_code=503, detail="Ultra precision detector not loaded")
    
    logger.info(f"🎯 Ultra precision corner detection request (budget: {time_budget}s)")
    
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        try:
            # Detect corners with ultra precision
            corners, time_taken, budget_met = ultra_detector.detect_corners(tmp_file_path, time_budget)
            
            if corners:
                logger.info(f"✅ Ultra precision successful: {time_taken:.3f}s (budget: {time_budget}s)")
                
                return JSONResponse(content={
                    "success": True,
                    "corners": corners,
                    "processing_time": round(time_taken, 3),
                    "time_budget": time_budget,
                    "budget_met": budget_met,
                    "accuracy_level": "ultra_precision",
                    "features_used": [
                        "multi_resolution_yolo",
                        "adaptive_subpixel", 
                        "intelligent_geometric",
                        "selective_edge" if time_taken > 0.3 else "skip_edge"
                    ]
                })
            else:
                logger.error("❌ Ultra precision detection failed")
                raise HTTPException(status_code=500, detail="Corner detection failed")
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"❌ Ultra precision detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/visualize_corners")
async def visualize_corners_endpoint(file: UploadFile = File(...), time_budget: float = Query(2.0, ge=0.5, le=10.0)):
    """
    Detect corners and return visualization image
    """
    if not ultra_detector:
        raise HTTPException(status_code=503, detail="Ultra precision detector not loaded")
    
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        try:
            # Load original image
            original_img = cv2.imread(tmp_file_path)
            if original_img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            
            # Detect corners
            corners, time_taken, budget_met = ultra_detector.detect_corners(tmp_file_path, time_budget)
            
            if corners:
                # Create visualization
                vis_img = self._create_ultra_precision_visualization(original_img, corners, time_taken, budget_met)
                
                # Encode to base64
                _, buffer = cv2.imencode('.jpg', vis_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return JSONResponse(content={
                    "success": True,
                    "corners": corners,
                    "processing_time": round(time_taken, 3),
                    "time_budget": time_budget,
                    "budget_met": budget_met,
                    "visualization": f"data:image/jpeg;base64,{img_base64}",
                    "accuracy_level": "ultra_precision"
                })
            else:
                raise HTTPException(status_code=500, detail="Corner detection failed")
                
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.post("/compare_accuracy")
async def compare_accuracy_endpoint(file: UploadFile = File(...)):
    """
    Compare ultra precision with other methods
    """
    if not ultra_detector:
        raise HTTPException(status_code=503, detail="Ultra precision detector not loaded")
    
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        try:
            # Test different time budgets
            budgets = [1.0, 1.5, 2.0]
            results = {}
            
            for budget in budgets:
                corners, time_taken, budget_met = ultra_detector.detect_corners(tmp_file_path, budget)
                
                results[f"budget_{budget}s"] = {
                    "corners": corners,
                    "time_taken": round(time_taken, 3),
                    "budget_met": budget_met,
                    "success": corners is not None
                }
            
            return JSONResponse(content={
                "success": True,
                "comparison_results": results,
                "recommendation": "Use 2.0s budget for maximum accuracy"
            })
            
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Accuracy comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Interactive demo page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultra Precision Corner Detection Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
            .error { background: #ffe6e6; border: 1px solid #ff9999; }
            .success { background: #e6ffe6; border: 1px solid #99ff99; }
            .performance { display: flex; justify-content: space-between; margin: 10px 0; }
            .metric { text-align: center; }
            img { max-width: 100%; height: auto; margin: 10px 0; }
            .budget-control { margin: 20px 0; }
            .budget-control input { width: 100px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Ultra Precision Corner Detection</h1>
            <p><strong>Port 8005</strong> - Maximum accuracy corner detection</p>
            <p>Target: &lt;15px error in &lt;2 seconds</p>
            
            <div class="budget-control">
                <label for="timeBudget">Time Budget (seconds):</label>
                <input type="number" id="timeBudget" min="0.5" max="10" step="0.1" value="2.0">
                <small>Recommended: 2.0s for maximum accuracy</small>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📸 Click to upload chess board image</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('fileInput').addEventListener('change', async function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const timeBudget = document.getElementById('timeBudget').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.style.display = 'block';
                resultDiv.className = 'result';
                resultDiv.innerHTML = '<p>🔄 Processing with ultra precision...</p>';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch(`/visualize_corners?time_budget=${timeBudget}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3>✅ Ultra Precision Detection Successful</h3>
                            <div class="performance">
                                <div class="metric">
                                    <strong>Processing Time</strong><br>
                                    ${data.processing_time}s
                                </div>
                                <div class="metric">
                                    <strong>Budget Met</strong><br>
                                    ${data.budget_met ? '✅ Yes' : '❌ No'}
                                </div>
                                <div class="metric">
                                    <strong>Accuracy Level</strong><br>
                                    Ultra Precision
                                </div>
                            </div>
                            <img src="${data.visualization}" alt="Corner detection result">
                            <details>
                                <summary>📊 Detection Details</summary>
                                <pre>${JSON.stringify(data.corners, null, 2)}</pre>
                                <p><strong>Features Used:</strong> ${data.features_used ? data.features_used.join(', ') : 'N/A'}</p>
                            </details>
                        `;
                    } else {
                        throw new Error('Detection failed');
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>❌ Detection Failed</h3><p>${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """

def _create_ultra_precision_visualization(image: np.ndarray, corners: List[List[float]], 
                                        time_taken: float, budget_met: bool) -> np.ndarray:
    """
    Create enhanced visualization showing ultra precision results
    """
    vis_img = image.copy()
    corners_np = np.array(corners, dtype=np.int32)
    
    # Enhanced corner visualization
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
    labels = ['TL', 'TR', 'BR', 'BL']
    
    # Draw corners with enhanced styling
    for i, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
        x, y = corner
        
        # Large corner markers
        cv2.circle(vis_img, (x, y), 20, color, -1)
        cv2.circle(vis_img, (x, y), 25, (255, 255, 255), 3)
        cv2.circle(vis_img, (x, y), 30, (0, 0, 0), 2)
        
        # Enhanced labels
        cv2.putText(vis_img, f'{label}', (x-25, y-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(vis_img, f'{label}', (x-25, y-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw quadrilateral with enhanced styling
    cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 4)
    cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 0, 0), 2)
    
    # Add performance information
    height, width = vis_img.shape[:2]
    
    # Background for text
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
    
    # Performance text
    budget_color = (0, 255, 0) if budget_met else (0, 0, 255)
    cv2.putText(vis_img, f"Ultra Precision Detection", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Time: {time_taken:.3f}s", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Budget: {'MET' if budget_met else 'EXCEEDED'}", (20, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, budget_color, 2)
    cv2.putText(vis_img, f"Target: <15px error", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_img

def main():
    """
    Start the Ultra Precision Corner Detection API
    """
    print("\n" + "="*60)
    print("Ultra Precision Corner Detection API")
    print("="*60)
    print("🛡️  SAFETY: Runs on port 8005 (separate from all other APIs)")
    print("   Main API (8001), YOLO API (8002), Precision API (8003), Fast Precision API (8004) unaffected")
    print("🚀 Starting Ultra Precision API on port 8005...")
    print("   Health check: http://localhost:8005/health")
    print("   API docs: http://localhost:8005/docs")
    print("   Demo: http://localhost:8005/demo")
    print("")
    print("🎯 ULTRA PRECISION FEATURES:")
    print("   ✅ Multi-resolution YOLO ensemble (640px + 896px)")
    print("   ✅ Adaptive sub-pixel refinement based on image quality")
    print("   ✅ Intelligent geometric optimization with confidence weighting")
    print("   ✅ Selective edge enhancement for uncertain corners")
    print("   ✅ Time budget management with graceful degradation")
    print("")
    print("🏆 TARGET: <15px average error in <2 seconds")
    print("🚀 Expected: 30-40% better than current Fast Precision (21.9px)")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8005)

if __name__ == "__main__":
    main()
