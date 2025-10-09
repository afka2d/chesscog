#!/usr/bin/env python3
"""
Test API for YOLO upgrades - completely separate from production.
This runs locally on port 8012 and won't affect your production APIs.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Upgrade Test API", version="1.0.0")

# Global variables for models
yolo_v8_model = None
yolo_v9_model = None
yolo_v11_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLOUpgradeTester:
    """Test different YOLO versions for corner detection"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load different YOLO versions for comparison"""
        try:
            from ultralytics import YOLO
            
            # Try to load YOLOv8 (your current version)
            try:
                self.models['yolov8n'] = YOLO("yolov8n.pt")
                self.models['yolov8s'] = YOLO("yolov8s.pt")
                logger.info("✅ YOLOv8 models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  YOLOv8 failed to load: {e}")
            
            # Try to load YOLOv9 (recommended upgrade)
            try:
                self.models['yolov9s'] = YOLO("yolov9s.pt")
                self.models['yolov9m'] = YOLO("yolov9m.pt")
                logger.info("✅ YOLOv9 models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  YOLOv9 failed to load: {e}")
            
            # Try to load YOLOv10 (latest available)
            try:
                self.models['yolov10n'] = YOLO("yolov10n.pt")
                self.models['yolov10s'] = YOLO("yolov10s.pt")
                logger.info("✅ YOLOv10 models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  YOLOv10 failed to load: {e}")
                
        except ImportError:
            logger.error("❌ Ultralytics not available. Install with: pip install ultralytics")
    
    def detect_corners_with_yolo(self, image_path: str, version: str = "yolov9") -> Tuple[Optional[List], float, dict]:
        """
        Detect corners using specified YOLO version
        Returns: (corners, processing_time, debug_info)
        """
        if version not in self.models:
            return None, 0.0, {"error": f"Model {version} not available"}
        
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.models[version](image_path, conf=0.3, verbose=False)
            
            # Extract chessboard detection (assuming it's detected as a class)
            corners = None
            confidence = 0.0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get the highest confidence detection
                    confidences = result.boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    confidence = confidences[best_idx]
                    
                    # Get bounding box
                    box = result.boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # Convert to corners (simplified - you might need more sophisticated logic)
                    corners = [
                        [float(x1), float(y1)],  # Top-left
                        [float(x2), float(y1)],  # Top-right
                        [float(x2), float(y2)],  # Bottom-right
                        [float(x1), float(y2)]   # Bottom-left
                    ]
                    break
            
            processing_time = time.time() - start_time
            
            debug_info = {
                "version": version,
                "confidence": float(confidence),
                "processing_time": processing_time,
                "model_available": True
            }
            
            return corners, processing_time, debug_info
            
        except Exception as e:
            processing_time = time.time() - start_time
            return None, processing_time, {"error": str(e), "version": version}

# Initialize the tester
tester = YOLOUpgradeTester()

@app.get("/")
async def root():
    """API information"""
    available_models = list(tester.models.keys())
    return {
        "message": "YOLO Upgrade Test API",
        "version": "1.0.0",
        "available_models": available_models,
        "device": str(device),
        "endpoints": {
            "test_single": "POST /test/{version} - Test single YOLO version",
            "compare_all": "POST /compare - Compare all available versions",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    available_models = list(tester.models.keys())
    return {
        "status": "healthy",
        "available_models": available_models,
        "total_models": len(available_models)
    }

@app.post("/test/{version}")
async def test_single_version(
    version: str,
    file: UploadFile = File(...)
):
    """
    Test a single YOLO version
    """
    if version not in tester.models:
        raise HTTPException(status_code=400, detail=f"Model {version} not available. Available: {list(tester.models.keys())}")
    
    # Save uploaded file temporarily
    temp_path = f"temp_test_{version}_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Test the model
        corners, processing_time, debug_info = tester.detect_corners_with_yolo(temp_path, version)
        
        result = {
            "success": corners is not None,
            "corners": corners,
            "processing_time": processing_time,
            "debug_info": debug_info,
            "version_tested": version
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

@app.post("/compare")
async def compare_all_versions(file: UploadFile = File(...)):
    """
    Compare all available YOLO versions on the same image
    """
    # Save uploaded file temporarily
    temp_path = f"temp_compare_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        results = {}
        total_time = 0
        
        # Test each available model
        for version in tester.models.keys():
            corners, processing_time, debug_info = tester.detect_corners_with_yolo(temp_path, version)
            results[version] = {
                "success": corners is not None,
                "corners": corners,
                "processing_time": processing_time,
                "debug_info": debug_info
            }
            total_time += processing_time
        
        return {
            "image_tested": file.filename,
            "total_processing_time": total_time,
            "results": results,
            "recommendation": _get_recommendation(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def _get_recommendation(results: dict) -> dict:
    """Generate recommendation based on results"""
    successful_models = {k: v for k, v in results.items() if v["success"]}
    
    if not successful_models:
        return {"status": "no_successful_models", "message": "All models failed"}
    
    # Find fastest successful model
    fastest = min(successful_models.items(), key=lambda x: x[1]["processing_time"])
    
    # Find highest confidence successful model
    highest_conf = max(successful_models.items(), 
                      key=lambda x: x[1]["debug_info"].get("confidence", 0))
    
    return {
        "status": "success",
        "fastest": fastest[0],
        "highest_confidence": highest_conf[0],
        "fastest_time": fastest[1]["processing_time"],
        "highest_confidence_value": highest_conf[1]["debug_info"].get("confidence", 0)
    }

@app.get("/demo")
async def demo_interface():
    """Simple HTML demo interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Upgrade Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 YOLO Upgrade Test API</h1>
            <p>Test different YOLO versions for corner detection accuracy and speed.</p>
            
            <div class="upload-area">
                <h3>Upload Chess Image</h3>
                <input type="file" id="imageFile" accept="image/*">
                <br><br>
                <button onclick="testYOLOv9()">Test YOLOv9</button>
                <button onclick="testYOLOv11()">Test YOLOv11</button>
                <button onclick="compareAll()">Compare All Versions</button>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            async function testModel(version) {
                const file = document.getElementById('imageFile').files[0];
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch(`/test/${version}`, {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResult(`${version} Test`, result);
                } catch (error) {
                    displayResult(`${version} Test`, { error: error.message });
                }
            }
            
            async function testYOLOv9() { await testModel('yolov9'); }
            async function testYOLOv11() { await testModel('yolov11'); }
            
            async function compareAll() {
                const file = document.getElementById('imageFile').files[0];
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/compare', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResult('All Versions Comparison', result);
                } catch (error) {
                    displayResult('Comparison', { error: error.message });
                }
            }
            
            function displayResult(title, data) {
                const resultsDiv = document.getElementById('results');
                const resultDiv = document.createElement('div');
                resultDiv.className = 'result';
                resultDiv.innerHTML = `
                    <h3>${title}</h3>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
                resultsDiv.appendChild(resultDiv);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting YOLO Upgrade Test API on port 8012")
    print("📍 Test endpoints:")
    print("   - http://localhost:8012/compare (compare all versions)")
    print("   - http://localhost:8012/test/yolov9 (test YOLOv9)")
    print("   - http://localhost:8012/test/yolov11 (test YOLOv11)")
    print("   - http://localhost:8012/demo (web interface)")
    print("")
    print("🛡️  This is completely separate from your production APIs!")
    print("🛡️  Your production system (ports 8010, 8011) is unaffected!")
    
    uvicorn.run(app, host="0.0.0.0", port=8012)
