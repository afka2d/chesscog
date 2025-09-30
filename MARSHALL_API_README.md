# Marshall Improved API

A new API endpoint that uses improved Marshall-trained models for better chess position recognition accuracy.

## 🎯 Overview

This API provides the same interface as the production API but uses improved models:
- **Marshall Occupancy Model**: Improved occupancy detection trained on Marshall Chess Club data
- **Original Color Model**: Keeps the working color classification model (no change needed)
- **Combined Piece Classifier**: New model trained on both Marshall and grey background data

## 🚀 Quick Start

### 1. Start the Marshall Improved API
```bash
python start_marshall_api.py
```

The API will start on **port 8003**.

### 2. Test the API
```bash
python test_marshall_api.py
```

### 3. Compare with Original API
```bash
python compare_apis.py
```

## 📍 API Endpoints

### Health Check
- **GET** `/health` - Check API health and model status
- **Example**: `curl http://localhost:8003/health`

### Debug Information
- **GET** `/debug/info` - Model information and configuration
- **Example**: `curl http://localhost:8003/debug/info`

### Chess Position Recognition
- **POST** `/recognize_chess_position_with_corners` - Main chess recognition endpoint
- **Parameters**:
  - `image`: Chess board image file (JPEG/PNG)
  - `corners`: JSON string of corner coordinates `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`
  - `debug`: Optional boolean for debug mode (`true`/`false`)
- **Returns**: Same format as production API

## 🔧 Model Details

### Occupancy Detection
- **Model**: Marshall-trained ResNet
- **Path**: `models_marshall_improved/occupancy_marshall.pt`
- **Improvement**: Better accuracy on both Marshall and grey background datasets

### Color Classification
- **Model**: Original MobileNetV2
- **Path**: `models/color_classifier_simple.pt`
- **Status**: Unchanged (working well)

### Piece Classification
- **Model**: Combined ResNet18
- **Path**: `models_marshall_improved/combined_piece_classifier.pt`
- **Training Data**: Marshall + Grey Background datasets
- **Improvement**: Better generalization across different chess sets

## 📊 Expected Improvements

Based on validation results:
- **Marshall Data**: 100% accuracy on Marshall chess set
- **Grey Background Data**: ~80% accuracy on grey background chess set
- **Combined Performance**: Better overall accuracy across different chess sets

## 🔄 Migration Guide

To switch from the original API to the Marshall Improved API:

### 1. Update API URL
```python
# Before
api_url = "http://localhost:8001"

# After  
api_url = "http://localhost:8003"
```

### 2. No Code Changes Required
The API maintains the exact same input/output format, so no other changes are needed.

## 🧪 Testing

### Test Individual API
```bash
python test_marshall_api.py
```

### Compare Both APIs
```bash
python compare_apis.py
```

### Manual Testing
```python
import requests
import json

# Test with sample image
corners = [[324, 324], [2916, 324], [2916, 5436], [324, 5436]]
files = {'image': open('chess_image.jpg', 'rb')}
data = {
    'corners': json.dumps(corners),
    'debug': 'true'
}

response = requests.post(
    'http://localhost:8003/recognize_chess_position_with_corners',
    files=files,
    data=data
)

result = response.json()
print(f"FEN: {result['fen']}")
print(f"Pieces detected: {sum(1 for p in result['pieces'] if p is not None)}")
```

## 🚨 Important Notes

1. **Port 8003**: This API runs on port 8003 to avoid conflicts with existing APIs
2. **Model Dependencies**: Ensure all required models are present before starting
3. **Same Interface**: Input/output format is identical to production API
4. **No Production Impact**: This API is completely separate from production systems

## 📁 File Structure

```
├── marshall_improved_api.py      # Main API implementation
├── start_marshall_api.py         # Launcher script
├── test_marshall_api.py          # Individual API testing
├── compare_apis.py               # Side-by-side comparison
├── MARSHALL_API_README.md        # This documentation
└── models_marshall_improved/     # Marshall-trained models
    ├── occupancy_marshall.pt
    └── combined_piece_classifier.pt
```

## 🔍 Troubleshooting

### API Won't Start
- Check that all required models exist
- Ensure port 8003 is available
- Check Python dependencies

### Poor Accuracy
- Verify models are loaded correctly
- Check debug output for confidence scores
- Ensure proper corner detection

### Model Loading Errors
- Verify model file paths
- Check model file permissions
- Ensure PyTorch compatibility

## 📈 Performance

- **Startup Time**: ~5-10 seconds (model loading)
- **Processing Time**: Similar to original API
- **Memory Usage**: Slightly higher due to larger models
- **Accuracy**: Improved across different chess sets
