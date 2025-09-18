# Local Development API Setup

This setup allows you to work on improving API accuracy locally without affecting your production app that's submitted to the App Store.

## 🎯 Overview

- **Production API**: Runs on port 8000, stable for App Store
- **Local Development API**: Runs on port 8001, for testing improvements
- **Separate Git Branch**: `local-development` branch for your work

## 🚀 Quick Start

### Start Local Development API
```bash
./start_local_dev.sh
```

This will:
- Start the API on port 8001
- Run a test to verify it's working
- Show you the available endpoints

### Manual Start
```bash
python main_local_dev.py
```

## 🔧 Development Features

The local development API includes additional features:

### 1. Debug Mode
Add `debug=true` to your API calls to get detailed information:
```python
data = {
    'corners': json.dumps(corners),
    'debug': 'true'  # Enable debug mode
}
```

### 2. Debug Endpoints
- `GET /debug/info` - Model information and configuration
- `GET /health` - Health check with environment info

### 3. Enhanced Logging
- Detailed confidence scores for each square
- Processing time information
- Step-by-step classification details

## 📊 Testing Your Improvements

### Test Script
```bash
python test_local_dev.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8001/health

# Debug info
curl http://localhost:8001/debug/info

# Full recognition with debug
curl -X POST http://localhost:8001/recognize_chess_position_with_corners \
  -F "image=@your_image.jpg" \
  -F "corners=[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]" \
  -F "debug=true"
```

## 🔄 Workflow

1. **Make improvements** to `main_local_dev.py`
2. **Test locally** using the test script or your app pointing to `localhost:8001`
3. **Iterate** until you're happy with the improvements
4. **Deploy to production** when ready (update production API)

## 🛠️ Making Improvements

### Common Areas to Improve

1. **Model Architecture**
   - Modify `_get_color_model_architecture()` or `_get_piece_type_model_architecture()`
   - Try different pre-trained models

2. **Confidence Thresholds**
   - Adjust the `0.7` threshold in the classification logic
   - Experiment with different thresholds for different piece types

3. **Image Preprocessing**
   - Modify the transform pipelines
   - Try different image sizes or augmentations

4. **Post-processing**
   - Add chess rule validation
   - Implement piece position validation

### Example: Lower Confidence Threshold
```python
# In main_local_dev.py, change this line:
if color_confidence >= 0.7 and piece_type_confidence >= 0.7:
# To:
if color_confidence >= 0.5 and piece_type_confidence >= 0.5:
```

## 📱 Testing with Your App

To test with your app, temporarily change the API URL in your app to:
```
http://localhost:8001/recognize_chess_position_with_corners
```

Remember to change it back to production before submitting updates!

## 🚀 Deploying Improvements

When you're ready to deploy improvements to production:

1. **Test thoroughly** on local development API
2. **Update production API** with your improvements
3. **Test production API** to ensure it works
4. **Update your app** to use production API

## 🔍 Debugging Tips

### Check Model Loading
```bash
curl http://localhost:8001/debug/info
```

### View Detailed Logs
The API logs detailed information about each square classification.

### Test with Different Images
Use various chess positions to test robustness.

## 📁 File Structure

```
├── main_local_dev.py          # Local development API
├── test_local_dev.py          # Test script
├── start_local_dev.sh         # Startup script
├── LOCAL_DEVELOPMENT_README.md # This file
└── main_final_piece_classifier.py # Production API (for reference)
```

## ⚠️ Important Notes

- **Never run both APIs on the same port**
- **Always test locally before deploying**
- **Keep production API stable during App Store review**
- **Use git branches to track your improvements**

## 🆘 Troubleshooting

### Port Already in Use
```bash
pkill -f "python main_local_dev.py"
```

### Models Not Loading
Check that model files exist:
- `runs/occupancy_classifier/ResNet/ResNet.pt`
- `models/color_classifier_simple.pt`
- `models/piece_classifier_simple.pt`

### API Not Responding
Check the logs for error messages and ensure all dependencies are installed.

---

Happy coding! 🎉
