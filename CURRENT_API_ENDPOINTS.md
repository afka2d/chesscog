# Current API Endpoints Available

## 🌐 **CURRENTLY RUNNING APIS**

Based on the running processes, here are your currently available API endpoints:

### **🎯 Main Chess Recognition API** (Port 8001)
**Status**: ✅ **RUNNING** (`main_local_dev.py`)
**URL**: `http://localhost:8001`

#### **Available Endpoints:**

#### **1. Health Check**
- **`GET /health`** - Check API health and model status
- **Example**: `curl http://localhost:8001/health`

#### **2. Debug Information**
- **`GET /debug/info`** - Model information and configuration
- **Example**: `curl http://localhost:8001/debug/info`

#### **3. Chess Position Recognition** 
- **`POST /recognize_chess_position_with_corners`** - Main chess recognition endpoint
- **Parameters**:
  - `image`: Chess board image file (JPEG/PNG)
  - `corners`: JSON string of corner coordinates `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`
  - `debug`: Optional boolean for debug mode (`true`/`false`)
- **Returns**:
  - `fen`: FEN notation of the chess position
  - `ascii`: ASCII representation of the board
  - `lichess_url`: Lichess editor URL for the position
  - `legal_position`: Whether the position is chess-legal
  - `pieces_detected`: Number of pieces detected
  - `processing_time`: Time taken for processing
  - `debug_info`: Detailed processing information (if debug=true)
  - `square_details`: Per-square classification details
  - `confidence_scores`: Model confidence for each classification

**Example Usage:**
```python
import requests
import json

# Prepare data
corners = [[324, 324], [2916, 324], [2916, 5436], [324, 5436]]
files = {'image': open('chess_image.jpg', 'rb')}
data = {
    'corners': json.dumps(corners),
    'debug': 'true'
}

# Make request
response = requests.post(
    'http://localhost:8001/recognize_chess_position_with_corners',
    files=files,
    data=data
)

result = response.json()
print(f"FEN: {result['fen']}")
print(f"Pieces detected: {result['pieces_detected']}")
```

---

## 🔧 **CORNER DETECTION APIS** (Background/Separate)

### **Corner Detection API** (Port 8002)
**Status**: ⚠️ **NOT CURRENTLY RUNNING** (port conflict detected)
**URL**: `http://localhost:8002` (when running)

#### **Available Endpoints** (when running):

#### **1. Root Information**
- **`GET /`** - API information and endpoint list

#### **2. Health Check**
- **`GET /health`** - Health check for corner detection service

#### **3. Corner Detection**
- **`POST /detect_corners`** - Detect chess board corners
- **Parameters**: 
  - `file`: Image file
- **Returns**: Corner coordinates

#### **4. Corner Visualization**
- **`POST /visualize_corners`** - Detect corners and return visualization
- **Parameters**: 
  - `file`: Image file  
- **Returns**: Base64-encoded image with corners drawn

#### **5. Demo Interface**
- **`GET /demo`** - Interactive HTML demo for corner detection

---

## 🚀 **PRODUCTION API** (Remote Server)

### **Production Chess Recognition API** (Port 8000)
**Status**: 🌍 **DEPLOYED** (remote server: `159.203.102.249`)
**URL**: `http://159.203.102.249:8000`

#### **Available Endpoints:**

#### **1. Root Information**
- **`GET /`** - API information and status

#### **2. Health Check**
- **`GET /health`** - Production health check

#### **3. Chess Position Recognition**
- **`POST /recognize_chess_position_with_corners`** - Production chess recognition
- **Same parameters and response as local development API**

---

## 🎯 **RECOMMENDED CORNER DETECTION SERVICES**

Based on our testing, here are the available corner detection methods:

### **1. Original Corner Detection Service**
```python
from corner_detection_service import CornerDetectionService
service = CornerDetectionService()
corners = service.detect_corners('image.jpg')
# Expected accuracy: 64.0 pixels average
```

### **2. Optimized Corner Detection Service** ⭐ **RECOMMENDED**
```python
from optimized_corner_service import OptimizedCornerService
service = OptimizedCornerService()
corners = service.detect_corners('image.jpg')
# Expected accuracy: 60.0 pixels average (6.3% better)
```

### **3. YOLO Corner Detection Service** 🏆 **BEST PERFORMANCE**
```python
from improved_yolo_corner_detection import ImprovedYOLOCornerDetector
service = ImprovedYOLOCornerDetector()
corners = service.detect_corners('image.jpg')
# Expected accuracy: 45.7 pixels average (28.6% better than original!)
```

---

## 📊 **PERFORMANCE COMPARISON**

| Service | Average Error | Improvement | Speed | Status |
|---------|--------------|-------------|-------|---------|
| **YOLO** 🏆 | **45.7 pixels** | **+28.6%** | ~0.2s | ✅ Available |
| **Optimized CNN** | 60.0 pixels | +6.3% | ~0.1s | ✅ Available |
| **Original CNN** | 64.0 pixels | Baseline | ~0.1s | ✅ Available |

---

## 🚀 **HOW TO USE THE BEST CORNER DETECTION**

### **Option 1: Use YOLO Service Directly**
```python
from improved_yolo_corner_detection import ImprovedYOLOCornerDetector

# Initialize YOLO service
yolo_service = ImprovedYOLOCornerDetector()

# Detect corners
corners = yolo_service.detect_corners('your_chess_image.jpg')

# Use corners with main API
import requests
import json

files = {'image': open('your_chess_image.jpg', 'rb')}
data = {'corners': json.dumps(corners)}

response = requests.post(
    'http://localhost:8001/recognize_chess_position_with_corners',
    files=files,
    data=data
)
```

### **Option 2: Integrated Workflow**
```python
from integrated_corner_detection_example import IntegratedChessRecognition

# Use the integrated service (combines YOLO + main API)
integrated = IntegratedChessRecognition()
result = integrated.process_image('your_chess_image.jpg')

print(f"FEN: {result['fen']}")
print(f"Pieces: {result['pieces_detected']}")
```

---

## 🔧 **TO START CORNER DETECTION API**

If you want to run the corner detection API on port 8002:

```bash
# Kill any existing process on port 8002
lsof -ti:8002 | xargs kill -9

# Start corner detection API
source venv/bin/activate && python corner_api_simple.py
```

---

## 📋 **SUMMARY**

**Currently Available**:
- ✅ **Main Chess API** (port 8001) - Working and ready
- ✅ **YOLO Corner Detection** - Best accuracy (45.7 pixels)
- ✅ **Optimized Corner Detection** - Good improvement (60.0 pixels)
- 🌍 **Production API** (port 8000, remote) - Deployed and stable

**Best Workflow**:
1. Use **YOLO corner detection** for automatic corners (most accurate)
2. Pass detected corners to **main chess API** for position recognition
3. Get complete chess position analysis with 28.6% better corner accuracy

**Your corner detection system now has multiple options, with YOLO providing the best performance!** 🚀
