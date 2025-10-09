#!/usr/bin/env python3
"""
Test API for custom YOLO model upgrades - using your actual chessboard-trained model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

app = FastAPI(title="Custom YOLO Upgrade Test API", version="1.0.0")

# Global variables for models
custom_yolo_model = None
standard_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomYOLOUpgradeTester:
    """Test different YOLO versions for corner detection using your custom model"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load different YOLO versions for comparison"""
        try:
            from ultralytics import YOLO
            
            # Load your custom chessboard-trained model
            custom_model_path = "yolo_training_runs/yolo_chessboard_v1/weights/best.pt"
            if Path(custom_model_path).exists():
                self.models['custom_chessboard'] = YOLO(custom_model_path)
                logger.info("✅ Custom chessboard YOLO loaded successfully")
            else:
                logger.warning(f"⚠️  Custom model not found: {custom_model_path}")
            
            # Load standard YOLO models for comparison
            standard_models = [
                ("yolov8n", "yolov8n.pt"),
                ("yolov8s", "yolov8s.pt"),
                ("yolov9s", "yolov9s.pt"),
                ("yolov9m", "yolov9m.pt"),
                ("yolov10n", "yolov10n.pt"),
                ("yolov10s", "yolov10s.pt")
            ]
            
            for name, model_path in standard_models:
                try:
                    if Path(model_path).exists():
                        self.models[name] = YOLO(model_path)
                        logger.info(f"✅ {name} loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️  {name} failed to load: {e}")
                    
        except ImportError:
            logger.error("❌ Ultralytics not available. Install with: pip install ultralytics")
    
    def detect_corners_with_custom_yolo(self, image_path: str) -> Tuple[Optional[List], float, dict]:
        """
        Detect corners using your custom chessboard-trained YOLO model
        """
        if 'custom_chessboard' not in self.models:
            return None, 0.0, {"error": "Custom chessboard model not available"}
        
        start_time = time.time()
        
        try:
            # Run custom YOLO detection
            results = self.models['custom_chessboard'](image_path, conf=0.3, verbose=False)
            
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
                    
                    # Convert to corners (your custom model should detect chessboard)
                    corners = [
                        [float(x1), float(y1)],  # Top-left
                        [float(x2), float(y1)],  # Top-right
                        [float(x2), float(y2)],  # Bottom-right
                        [float(x1), float(y2)]   # Bottom-left
                    ]
                    break
            
            processing_time = time.time() - start_time
            
            debug_info = {
                "version": "custom_chessboard",
                "confidence": float(confidence),
                "processing_time": processing_time,
                "model_type": "custom_trained_chessboard"
            }
            
            return corners, processing_time, debug_info
            
        except Exception as e:
            processing_time = time.time() - start_time
            return None, processing_time, {"error": str(e), "version": "custom_chessboard"}
    
    def detect_corners_with_standard_yolo(self, image_path: str, version: str) -> Tuple[Optional[List], float, dict]:
        """
        Detect corners using standard YOLO models (won't work well for chessboards)
        """
        if version not in self.models:
            return None, 0.0, {"error": f"Model {version} not available"}
        
        start_time = time.time()
        
        try:
            # Run standard YOLO detection
            results = self.models[version](image_path, conf=0.3, verbose=False)
            
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
                    
                    # Convert to corners
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
                "model_type": "standard_pretrained"
            }
            
            return corners, processing_time, debug_info
            
        except Exception as e:
            processing_time = time.time() - start_time
            return None, processing_time, {"error": str(e), "version": version}

# Initialize the tester
tester = CustomYOLOUpgradeTester()

@app.get("/")
async def root():
    """API information"""
    available_models = list(tester.models.keys())
    return {
        "message": "Custom YOLO Upgrade Test API",
        "version": "1.0.0",
        "available_models": available_models,
        "device": str(device),
        "endpoints": {
            "test_custom": "POST /test_custom - Test your custom chessboard model",
            "test_standard": "POST /test_standard/{version} - Test standard YOLO models",
            "compare": "POST /compare - Compare custom vs standard models",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    available_models = list(tester.models.keys())
    custom_available = 'custom_chessboard' in available_models
    return {
        "status": "healthy",
        "available_models": available_models,
        "total_models": len(available_models),
        "custom_chessboard_model": custom_available
    }

@app.post("/test_custom")
async def test_custom_model(file: UploadFile = File(...)):
    """
    Test your custom chessboard-trained YOLO model
    """
    # Save uploaded file temporarily
    temp_path = f"temp_custom_test_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Test the custom model
        corners, processing_time, debug_info = tester.detect_corners_with_custom_yolo(temp_path)
        
        result = {
            "success": corners is not None,
            "corners": corners,
            "processing_time": processing_time,
            "debug_info": debug_info,
            "model_tested": "custom_chessboard"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

@app.post("/test_standard/{version}")
async def test_standard_model(version: str, file: UploadFile = File(...)):
    """
    Test a standard YOLO model (for comparison)
    """
    if version not in tester.models:
        raise HTTPException(status_code=400, detail=f"Model {version} not available. Available: {list(tester.models.keys())}")
    
    # Save uploaded file temporarily
    temp_path = f"temp_standard_test_{version}_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Test the standard model
        corners, processing_time, debug_info = tester.detect_corners_with_standard_yolo(temp_path, version)
        
        result = {
            "success": corners is not None,
            "corners": corners,
            "processing_time": processing_time,
            "debug_info": debug_info,
            "model_tested": version
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Compare your custom model with standard YOLO models
    """
    # Save uploaded file temporarily
    temp_path = f"temp_compare_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        results = {}
        total_time = 0
        
        # Test custom model first
        if 'custom_chessboard' in tester.models:
            corners, processing_time, debug_info = tester.detect_corners_with_custom_yolo(temp_path)
            results['custom_chessboard'] = {
                "success": corners is not None,
                "corners": corners,
                "processing_time": processing_time,
                "debug_info": debug_info,
                "model_type": "custom_trained"
            }
            total_time += processing_time
        
        # Test a few standard models for comparison
        standard_models_to_test = ['yolov8n', 'yolov9s', 'yolov10n']
        for version in standard_models_to_test:
            if version in tester.models:
                corners, processing_time, debug_info = tester.detect_corners_with_standard_yolo(temp_path, version)
                results[version] = {
                    "success": corners is not None,
                    "corners": corners,
                    "processing_time": processing_time,
                    "debug_info": debug_info,
                    "model_type": "standard_pretrained"
                }
                total_time += processing_time
        
        return {
            "image_tested": file.filename,
            "total_processing_time": total_time,
            "results": results,
            "recommendation": _get_custom_recommendation(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def _get_custom_recommendation(results: dict) -> dict:
    """Generate recommendation based on custom vs standard model results"""
    custom_result = results.get('custom_chessboard')
    
    if not custom_result:
        return {"status": "no_custom_model", "message": "Custom chessboard model not available"}
    
    if custom_result['success']:
        custom_conf = custom_result['debug_info'].get('confidence', 0)
        custom_time = custom_result['processing_time']
        
        # Find best standard model
        standard_results = {k: v for k, v in results.items() if k != 'custom_chessboard' and v['success']}
        
        if standard_results:
            best_standard = max(standard_results.items(), 
                              key=lambda x: x[1]['debug_info'].get('confidence', 0))
            best_standard_conf = best_standard[1]['debug_info'].get('confidence', 0)
            
            if custom_conf > best_standard_conf:
                improvement = ((custom_conf - best_standard_conf) / best_standard_conf) * 100 if best_standard_conf > 0 else 100
                return {
                    "status": "custom_wins",
                    "message": f"Your custom model is {improvement:.1f}% better than standard models",
                    "custom_confidence": custom_conf,
                    "best_standard_confidence": best_standard_conf,
                    "recommendation": "Keep using your custom model - it's specifically trained for chessboards"
                }
            else:
                return {
                    "status": "standard_better",
                    "message": "Standard models performed better (unexpected for chessboards)",
                    "custom_confidence": custom_conf,
                    "best_standard_confidence": best_standard_conf,
                    "recommendation": "Investigate why standard models are better"
                }
        else:
            return {
                "status": "custom_only_success",
                "message": "Only your custom model succeeded",
                "custom_confidence": custom_conf,
                "recommendation": "Your custom model is essential for chessboard detection"
            }
    else:
        return {"status": "custom_failed", "message": "Custom model failed - check model file"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Custom YOLO Upgrade Test API on port 8013")
    print("📍 Test endpoints:")
    print("   - http://localhost:8013/test_custom (test your custom model)")
    print("   - http://localhost:8013/compare (compare custom vs standard)")
    print("   - http://localhost:8013/health (health check)")
    print("")
    print("🛡️  This tests YOUR custom chessboard-trained model!")
    print("🛡️  Your production system (ports 8010, 8011) is unaffected!")
    
    uvicorn.run(app, host="0.0.0.0", port=8013)

