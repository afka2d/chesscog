#!/usr/bin/env python3
"""
Fast Precision Corner Detection API
====================================

Balanced corner detection API optimizing for speed while maintaining accuracy.
Runs on port 8004 to avoid conflicts with existing APIs.

Features:
- Target: Under 3 seconds per image
- Improved accuracy over YOLO-only
- Lightweight refinement pipeline
- Time budget management
- Graceful degradation when time runs out
"""

import os
import sys
import io
import base64
import logging
import traceback
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our fast precision detector
try:
    from fast_precision_detector import FastPrecisionDetector
    FAST_PRECISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fast precision detector not available: {e}")
    FAST_PRECISION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fast Precision Chess Corner Detection API",
    description="Balanced corner detection optimizing speed and accuracy",
    version="1.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
fast_detector: Optional[FastPrecisionDetector] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the fast precision corner detection system"""
    global fast_detector
    
    print("Fast Precision Corner Detection API")
    print("=" * 60)
    print("🛡️  SAFETY: Runs on port 8004 (separate from all other APIs)")
    print("   Main API (8001), YOLO API (8002), Precision API (8003) unaffected")
    print("🚀 Starting Fast Precision API on port 8004...")
    print("   Health check: http://localhost:8004/health")
    print("   API docs: http://localhost:8004/docs")
    print("   Demo: http://localhost:8004/demo")
    
    if not FAST_PRECISION_AVAILABLE:
        logger.error("❌ Fast precision detector not available")
        fast_detector = None
        return
    
    try:
        logger.info("🔧 Loading fast precision detection system...")
        fast_detector = FastPrecisionDetector()
        logger.info("✅ Fast precision detection system loaded successfully")
        print("✅ Fast precision detection system loaded successfully")
        print("🎯 Target: Under 3 seconds per image")
        print("⚡ Pipeline: YOLO → Sub-pixel → Geometric → (Optional Edge)")
        print("🏆 Expected: Better accuracy than YOLO, much faster than full precision")
        
    except Exception as e:
        logger.error(f"❌ Failed to load fast precision detector: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"❌ Failed to load fast precision detector: {e}")
        fast_detector = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fast Precision Chess Corner Detection API",
        "version": "1.5.0",
        "model": "Optimized YOLO + Lightweight Refinement",
        "target_speed": "Under 3 seconds per image",
        "pipeline_stages": [
            "1. YOLO initial detection (~0.1s)",
            "2. Fast sub-pixel refinement (~0.1s)", 
            "3. Lightweight geometric validation (~0.1s)",
            "4. Optional edge refinement (if time permits)"
        ],
        "optimization": "Balanced speed vs accuracy",
        "endpoints": {
            "/detect_corners": "POST - Fast precision corner detection",
            "/detect_corners_with_budget": "POST - Detection with custom time budget",
            "/visualize_corners": "POST - Detection with visualization",
            "/compare_speeds": "POST - Compare with other methods",
            "/health": "GET - Health check",
            "/demo": "GET - Interactive demo"
        },
        "detector_loaded": fast_detector is not None,
        "safety_note": "Runs on port 8004 - independent of all other APIs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if fast_detector is not None else "detector_not_loaded"
    return {
        "status": status,
        "detector_loaded": fast_detector is not None,
        "pipeline": "YOLO + Fast Sub-pixel + Lightweight Geometric + Optional Edge",
        "target_speed": "Under 3 seconds",
        "port": 8004,
        "service": "Fast Precision Corner Detection"
    }

@app.post("/detect_corners")
async def detect_corners_fast_precision(file: UploadFile = File(...), time_budget: float = Query(3.0, ge=0.5, le=10.0)):
    """
    Fast precision corner detection with time budget
    
    Args:
        file: Chess board image
        time_budget: Maximum processing time in seconds (default: 3.0)
    
    Returns:
        - corners: Detected corner coordinates
        - processing_time: Actual processing time
        - time_budget_met: Whether detection completed within budget
        - pipeline_stages: Which stages were completed
    """
    if fast_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Fast precision corner detection system not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Detect corners with time budget
            logger.info(f"🚀 Starting fast precision detection (budget: {time_budget}s)...")
            corners = fast_detector.detect_corners_fast_precision(tmp_file_path, time_budget)
            processing_time = time.time() - start_time
            
            if corners is None:
                raise HTTPException(
                    status_code=422,
                    detail="Could not detect chess board corners with fast precision"
                )
            
            # Convert to proper format
            if hasattr(corners, 'tolist'):
                corners = corners.tolist()
            elif isinstance(corners, np.ndarray):
                corners = corners.tolist()
            
            time_budget_met = processing_time <= time_budget
            
            logger.info(f"✅ Fast precision successful: {processing_time:.3f}s (budget: {time_budget}s)")
            
            return {
                "corners": corners,
                "processing_time": round(processing_time, 3),
                "time_budget": time_budget,
                "time_budget_met": time_budget_met,
                "pipeline": "YOLO + Fast Sub-pixel + Lightweight Geometric + Optional Edge",
                "speed_class": "fast_precision",
                "api_version": "1.5.0"
            }
            
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fast precision detection error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Fast precision detection failed: {str(e)}")

@app.post("/detect_corners_with_budget")
async def detect_corners_custom_budget(file: UploadFile = File(...), time_budget: float = Query(..., ge=0.1, le=30.0)):
    """
    Corner detection with custom time budget
    
    Allows fine-tuning the speed vs accuracy trade-off
    """
    return await detect_corners_fast_precision(file, time_budget)

@app.post("/visualize_corners")
async def visualize_fast_precision_corners(file: UploadFile = File(...), time_budget: float = Query(3.0, ge=0.5, le=10.0)):
    """
    Fast precision corner detection with visualization
    """
    if fast_detector is None:
        raise HTTPException(status_code=503, detail="Fast precision detector not loaded")
    
    try:
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load original image
            original_img = cv2.imread(tmp_file_path)
            if original_img is None:
                raise HTTPException(status_code=422, detail="Could not load image")
            
            # Detect corners
            corners = fast_detector.detect_corners_fast_precision(tmp_file_path, time_budget)
            processing_time = time.time() - start_time
            
            if corners is None:
                raise HTTPException(status_code=422, detail="Fast precision corner detection failed")
            
            # Create visualization
            vis_img = original_img.copy()
            corners_np = np.array(corners, dtype=np.int32)
            
            # Draw fast precision corners with speed-focused styling
            colors = [(255, 165, 0), (0, 255, 255), (255, 20, 147), (50, 205, 50)]  # Orange, Cyan, Deep Pink, Lime
            labels = ['TL', 'TR', 'BR', 'BL']
            
            for i, (corner, color, label) in enumerate(zip(corners_np, colors, labels)):
                x, y = int(corner[0]), int(corner[1])
                
                # Draw speed-optimized markers
                cv2.circle(vis_img, (x, y), 18, color, -1)  # Filled circle
                cv2.circle(vis_img, (x, y), 22, (255, 255, 255), 2)  # White border
                cv2.circle(vis_img, (x, y), 4, (0, 0, 0), -1)  # Center dot
                
                # Label
                cv2.putText(vis_img, label, (x-12, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis_img, label, (x-12, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            # Draw quadrilateral
            cv2.polylines(vis_img, [corners_np.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
            
            # Add speed-focused title and info
            time_status = "⚡ FAST" if processing_time <= time_budget else "⏰ SLOW"
            cv2.putText(vis_img, f'{time_status} Precision Detection', (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(vis_img, f'Speed-Optimized Pipeline - {processing_time:.3f}s/{time_budget:.1f}s', (30, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', vis_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            time_budget_met = processing_time <= time_budget
            
            return {
                "corners": corners,
                "image": img_base64,
                "processing_time": round(processing_time, 3),
                "time_budget": time_budget,
                "time_budget_met": time_budget_met,
                "pipeline": "YOLO + Fast Sub-pixel + Lightweight Geometric + Optional Edge",
                "speed_class": "fast_precision",
                "image_format": "base64_jpeg"
            }
            
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fast precision visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.post("/compare_speeds")
async def compare_detection_speeds(file: UploadFile = File(...)):
    """
    Compare fast precision with YOLO-only detection
    
    Returns timing and accuracy comparison
    """
    if fast_detector is None:
        raise HTTPException(status_code=503, detail="Fast precision detector not loaded")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Test YOLO-only
            yolo_start = time.time()
            yolo_corners = fast_detector._fast_yolo_detection(tmp_file_path)
            yolo_time = time.time() - yolo_start
            
            # Test fast precision
            fast_start = time.time()
            fast_corners = fast_detector.detect_corners_fast_precision(tmp_file_path, 3.0)
            fast_time = time.time() - fast_start
            
            return {
                "yolo_only": {
                    "corners": yolo_corners,
                    "processing_time": round(yolo_time, 3),
                    "method": "YOLO-only"
                },
                "fast_precision": {
                    "corners": fast_corners,
                    "processing_time": round(fast_time, 3),
                    "method": "YOLO + Fast Refinement"
                },
                "comparison": {
                    "speed_ratio": round(fast_time / yolo_time, 1) if yolo_time > 0 else None,
                    "both_successful": yolo_corners is not None and fast_corners is not None,
                    "recommendation": "fast_precision" if fast_time < 3.0 else "yolo_only"
                }
            }
            
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Speed comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Speed comparison failed: {str(e)}")

@app.get("/demo", response_class=HTMLResponse)
async def fast_precision_demo_page():
    """Interactive demo page for fast precision corner detection"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fast Precision Corner Detection Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 3px dashed #ff6b35; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .result { margin: 20px 0; }
            img { max-width: 100%; height: auto; border-radius: 8px; }
            .info { background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0; }
            .pipeline { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
            .stage { display: inline-block; margin: 5px; padding: 8px 12px; background: #e9ecef; border-radius: 5px; }
            .speed-control { margin: 20px 0; }
            .speed-slider { width: 100%; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>⚡ Fast Precision Chess Corner Detection</h1>
        
        <div class="info">
            <h3>🚀 Speed-Optimized Precision!</h3>
            <p><strong>Target:</strong> Under 3 seconds per image</p>
            <p><strong>Pipeline:</strong> YOLO → Sub-pixel → Geometric → Optional Edge</p>
            <p><strong>Benefit:</strong> Better accuracy than YOLO, much faster than full precision</p>
            <p><strong>Port:</strong> 8004 (completely separate from other APIs)</p>
        </div>
        
        <div class="pipeline">
            <h4>⚡ Fast Pipeline Stages:</h4>
            <div class="stage">1️⃣ YOLO (~0.1s)</div>
            <div class="stage">2️⃣ Sub-pixel (~0.1s)</div>
            <div class="stage">3️⃣ Geometric (~0.1s)</div>
            <div class="stage">4️⃣ Edge (if time)</div>
        </div>
        
        <div class="speed-control">
            <h4>⏱️ Time Budget Control:</h4>
            <label for="timeBudget">Time Budget: <span id="timeBudgetValue">3.0</span> seconds</label>
            <input type="range" id="timeBudget" class="speed-slider" min="0.5" max="10" step="0.5" value="3.0">
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>📁 Click here to select a chess board image for fast precision corner detection</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div id="result" class="result"></div>
        
        <script>
            // Update time budget display
            document.getElementById('timeBudget').addEventListener('input', function(e) {
                document.getElementById('timeBudgetValue').textContent = e.target.value;
            });
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const timeBudget = document.getElementById('timeBudget').value;
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('result').innerHTML = `<p>⚡ Running fast precision detection (${timeBudget}s budget)...</p>`;
                
                fetch(`/visualize_corners?time_budget=${timeBudget}`, {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.image) {
                        const budgetStatus = data.time_budget_met ? '⚡ Within Budget' : '⏰ Over Budget';
                        const speedClass = data.time_budget_met ? 'success' : 'warning';
                        
                        document.getElementById('result').innerHTML = `
                            <h3>✅ Fast Precision Detection Results</h3>
                            <p><strong>Processing time:</strong> ${data.processing_time}s / ${data.time_budget}s</p>
                            <p><strong>Budget status:</strong> ${budgetStatus}</p>
                            <p><strong>Pipeline:</strong> ${data.pipeline}</p>
                            <p><strong>Speed class:</strong> ${data.speed_class}</p>
                            <p><strong>Corners:</strong></p>
                            <pre>${JSON.stringify(data.corners, null, 2)}</pre>
                            <img src="data:image/jpeg;base64,${data.image}" alt="Fast Precision Corner Detection Result">
                        `;
                    } else {
                        document.getElementById('result').innerHTML = '<p>❌ Error: ' + (data.detail || 'Unknown error') + '</p>';
                    }
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = '<p>❌ Error: ' + error.message + '</p>';
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("\n🚀 Starting Fast Precision Corner Detection API...")
    print("🛡️  SAFETY: This runs on port 8004, separate from all other APIs")
    print("⚡ Optimized for speed while maintaining accuracy improvement")
    print("🎯 Target: Under 3 seconds per image")
    print("\n📍 Endpoints:")
    print("   Health: http://localhost:8004/health")
    print("   Docs: http://localhost:8004/docs") 
    print("   Demo: http://localhost:8004/demo")
    print("\n" + "="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
