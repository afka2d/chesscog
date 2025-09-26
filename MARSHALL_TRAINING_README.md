# Marshall Training System

This system creates improved versions of your chess recognition models using the Marshall training data, without affecting your current working models.

## 🎯 What This Does

- **Trains 4 improved models** using your Marshall annotations:
  - Corner Detection (Marshall improved)
  - Occupancy Detection (Marshall improved) 
  - Color Classification (Marshall improved)
  - Piece Classification (Marshall improved)

- **Preserves all existing models** - your current working API remains untouched
- **Creates new API endpoints** on port 8006 for the improved models
- **Uses your 523+ Marshall annotations** for training

## 📁 File Structure

```
models_marshall_improved/          # New improved models (safe)
├── corner_detection_marshall.pt
├── occupancy_marshall.pt
├── color_classification_marshall.pt
├── piece_classification_marshall.pt
└── model_info.json

marshall_training_pipeline.py      # Training pipeline
marshall_improved_api.py           # New API (port 8006)
train_marshall_models.py           # Training script
compare_apis.py                    # Compare original vs improved
```

## 🚀 Quick Start

### 1. Train the Models

```bash
python train_marshall_models.py
```

This will:
- Load your Marshall annotations
- Create training datasets
- Train all 4 improved models
- Save models to `models_marshall_improved/`

### 2. Start the Improved API

```bash
python marshall_improved_api.py
```

This starts the Marshall Improved API on port 8006.

### 3. Test Both APIs

```bash
python compare_apis.py
```

This compares performance between your original API (port 8001) and the improved API (port 8006).

## 🔗 API Endpoints

The Marshall Improved API (port 8006) provides:

- `POST /detect_corners` - Improved corner detection
- `POST /analyze_position` - Full position analysis
- `POST /visualize_corners` - Visualize detected corners
- `GET /health` - Health check

## 🛡️ Safety Features

- **No impact on current models** - all existing files remain unchanged
- **Separate model directory** - `models_marshall_improved/`
- **Separate API port** - 8006 (vs 8001 for original)
- **Backup created** - original models are never modified

## 📊 Training Data

- **Total annotations**: 523+ Marshall photos
- **Excluded images**: 48 (horizontal/problematic)
- **Chess set**: Marshall (different from original training)
- **Data augmentation**: Built into training pipeline

## 🔧 Model Architectures

- **Corner Detection**: EfficientNet-B0 + custom regressor
- **Occupancy**: ResNet18 (2 classes: occupied/empty)
- **Color**: MobileNetV2 (2 classes: white/black)
- **Piece**: EfficientNet-B0 (6 classes: pawn/knight/bishop/rook/queen/king)

## 📈 Expected Improvements

The Marshall models should show improvements in:
- **Accuracy on Marshall-style boards**
- **Robustness to different lighting**
- **Better corner detection on similar boards**
- **Improved piece recognition on Marshall set**

## 🚨 Important Notes

1. **Your current API (port 8001) is completely safe** - no changes made
2. **Training takes time** - expect 30-60 minutes for all models
3. **GPU recommended** - training will be much faster with CUDA
4. **Test thoroughly** - compare both APIs before switching

## 🔄 Switching to Improved Models

When you're ready to use the improved models:

1. **Test both APIs** with `compare_apis.py`
2. **Verify improved performance** on your use cases
3. **Update your main API** to use the Marshall models (optional)
4. **Keep backups** of your original models

## 🐛 Troubleshooting

### Training fails
- Check Marshall annotations exist: `marshall_chess_annotations/annotations.json`
- Check Marshall photos exist: `/Users/tonyblum/Desktop/marshall photos`
- Ensure enough disk space for models

### API won't start
- Check models were trained successfully
- Verify model files exist in `models_marshall_improved/`
- Check port 8006 is available

### Poor performance
- Marshall models are trained on Marshall data
- May not perform as well on very different chess sets
- Consider retraining with more diverse data

## 📞 Support

If you encounter issues:
1. Check the logs for error messages
2. Verify all file paths are correct
3. Ensure sufficient training data
4. Test with a small subset first

---

**Remember**: Your original working models are completely safe and unchanged! 🛡️

