# 🎯 Quick Training Guide for Your Chess Recognition Models

## 📊 Your Current Dataset
- **14 chess board images** ready for training
- **2 images** have sample FEN annotations
- **All images** have annotated versions to help with corner coordinates

## 🚀 Quick Start Options

### Option 1: Train with Current Data (Recommended)
```bash
# Train models with the data we have
python batch_train_models.py --train
```

### Option 2: Complete Pipeline (Best Results)
```bash
# Run the full pipeline: update FEN, validate, train
python batch_train_models.py --full-pipeline
```

## 📝 Manual Annotation Process

### Step 1: Review Annotated Images
Look at the annotated images in `custom_training_data/annotations/`:
- Red dots show corner points
- Grid overlay helps identify coordinates
- Use these to update corner coordinates in JSON files

### Step 2: Update Corner Coordinates
For each image, update the `corners` array in the JSON file:
```json
"corners": [
  [x1, y1],  // Top-left corner of chess board
  [x2, y2],  // Top-right corner of chess board  
  [x3, y3],  // Bottom-right corner of chess board
  [x4, y4]   // Bottom-left corner of chess board
]
```

### Step 3: Add FEN Notations
For each image, add the correct FEN notation:
```json
"fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1"
```

## 🎯 Sample FEN Notations for Common Positions

### Empty Board
```json
"fen": "8/8/8/8/8/8/8/8 w - - 0 1"
```

### Two Pawns (like IMG_4698)
```json
"fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1"
```

### Starting Position
```json
"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### Common Midgame Position
```json
"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
```

## 🔧 Training Commands

### Check Dataset Status
```bash
python batch_train_models.py --summary
```

### Validate Annotations
```bash
python batch_train_models.py --validate
```

### Train Models
```bash
python batch_train_models.py --train
```

### Full Pipeline
```bash
python batch_train_models.py --full-pipeline
```

## 📁 File Structure
```
custom_training_data/
├── images/                    # Your chess board photos
│   ├── IMG_4540.jpeg
│   ├── IMG_4698.JPG
│   └── ... (14 total images)
├── annotations/               # JSON annotation files
│   ├── IMG_4540.json
│   ├── IMG_4698.json
│   └── ... (14 total files)
└── annotations/               # Annotated images for reference
    ├── IMG_4540_annotated.jpg
    ├── IMG_4698_annotated.jpg
    └── ... (14 total files)
```

## 🎯 Next Steps

1. **Quick Training**: Run `python batch_train_models.py --train` to train with current data
2. **Manual Annotation**: Update corner coordinates and FEN notations for better results
3. **Full Pipeline**: Run `python batch_train_models.py --full-pipeline` for complete training

## 💡 Tips for Better Results

- **Corner Accuracy**: Precise corner coordinates are crucial for good results
- **FEN Accuracy**: Correct FEN notation ensures proper training labels
- **Image Variety**: Your 14 images provide good variety for training
- **Validation**: Always validate annotations before training

## 🚀 Ready to Train?

Your dataset is ready for training! Start with:

```bash
python batch_train_models.py --train
```

This will train both occupancy and piece classifiers on your custom chess board images. 