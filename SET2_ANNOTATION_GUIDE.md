# Chess Set 2 Annotation Guide

## Overview

This guide helps you annotate your new chess set images (Set 2) with corners and FEN positions. The system uses the robust corner detection API to provide initial corner estimates, which you can then manually adjust as needed.

## Quick Start

### 1. Setup
```bash
# Make sure the robust corner detection API is running
python robust_corner_api.py &

# Run the setup script
python setup_set2_annotation.py
```

### 2. Choose Your Annotation Method

#### Option A: Interactive Annotation (Recommended for first time)
```bash
python semi_automated_annotation_tool.py
```
- Process one image at a time
- Full control over each step
- Good for learning the process

#### Option B: Batch Processing (For many images)
```bash
python batch_annotation_helper.py "/path/to/your/images" --output "./chess_set2_annotations"
```
- Process multiple images efficiently
- Resume capability
- Progress tracking

## Detailed Usage

### Interactive Annotation Tool

The `semi_automated_annotation_tool.py` provides step-by-step annotation:

1. **Load Image**: Automatically loads the next image
2. **Auto-Detect Corners**: Uses robust corner detection API
3. **Review Corners**: Shows detected corners on the image
4. **Adjust if Needed**: Click to manually adjust corners
5. **Enter FEN**: Input the chess position
6. **Save Annotation**: Automatically saves to JSON file

**Controls:**
- `y` - Accept auto-detected corners
- `m` - Manually adjust corners
- `s` - Skip this image
- `q` - Quit processing

### Batch Annotation Helper

The `batch_annotation_helper.py` processes multiple images efficiently:

```bash
# Basic usage
python batch_annotation_helper.py "/path/to/images" --output "./annotations"

# Process only first 10 images
python batch_annotation_helper.py "/path/to/images" --output "./annotations" --max 10

# Resume from image 20
python batch_annotation_helper.py "/path/to/images" --output "./annotations" --start 20

# Specify chess set
python batch_annotation_helper.py "/path/to/images" --output "./annotations" --chess-set "set2"
```

**Features:**
- Progress tracking
- Resume capability
- Quality control
- Statistics and reporting
- Automatic visualization generation

## Output Structure

```
chess_set2_annotations/
├── annotations/           # JSON annotation files
│   ├── IMG_0001.json
│   ├── IMG_0002.json
│   └── ...
├── visualizations/        # Corner visualization images
│   ├── IMG_0001_corners.jpg
│   ├── IMG_0002_corners.jpg
│   └── ...
├── reports/              # Processing reports
│   └── annotation_report_20241201_143022.json
├── progress.json         # Progress tracking
└── statistics.json       # Processing statistics
```

## Annotation Format

Each annotation file contains:

```json
{
  "image_path": "/path/to/image.jpg",
  "image_name": "IMG_0001.jpg",
  "chess_set": "set2",
  "corners": [
    [x1, y1],  // Top-left
    [x2, y2],  // Top-right
    [x3, y3],  // Bottom-right
    [x4, y4]   // Bottom-left
  ],
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "annotation_method": "semi_automated_batch",
  "corner_detection_api": "robust_port_8005",
  "timestamp": "2024-12-01T14:30:22.123456"
}
```

## FEN Format

The FEN (Forsyth-Edwards Notation) format represents the chess position:

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

**Components:**
- `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` - Board position
- `w` - Active color (w=white, b=black)
- `KQkq` - Castling rights
- `-` - En passant square
- `0` - Halfmove clock
- `1` - Fullmove number

**Piece Notation:**
- `K` - White King
- `Q` - White Queen
- `R` - White Rook
- `B` - White Bishop
- `N` - White Knight
- `P` - White Pawn
- `k` - Black King
- `q` - Black Queen
- `r` - Black Rook
- `b` - Black Bishop
- `n` - Black Knight
- `p` - Black Pawn
- `1-8` - Empty squares (number = count)

## Tips for Accurate Annotation

### Corner Detection
1. **Review Auto-Detected Corners**: The robust API usually provides good initial corners
2. **Manual Adjustment**: Click on corners to adjust if needed
3. **Precision Matters**: Small corner errors cause large board warping errors
4. **Consistent Order**: Always use TL, TR, BR, BL order

### FEN Input
1. **Use Chess Notation**: Standard algebraic notation
2. **Include All Components**: Board, active color, castling, en passant, clocks
3. **Validate FEN**: The system checks FEN validity
4. **Default Ending**: If you only provide the board, ` w KQkq - 0 1` is added automatically

### Image Quality
1. **Good Lighting**: Ensure the board is well-lit
2. **Clear View**: Avoid shadows and reflections
3. **Stable Position**: Keep the camera steady
4. **Full Board**: Include the entire chessboard in the frame

## Troubleshooting

### Common Issues

**1. Corner Detection API Not Available**
```
❌ Robust corner detection API not available
```
**Solution:** Start the API first:
```bash
python robust_corner_api.py &
```

**2. Images Not Found**
```
❌ No chess images found in common locations
```
**Solution:** Specify the correct path:
```bash
python batch_annotation_helper.py "/correct/path/to/images"
```

**3. Invalid FEN Format**
```
❌ Invalid FEN. Please check format.
```
**Solution:** Use standard chess notation:
- `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

**4. Corner Adjustment Issues**
```
❌ Must provide exactly 4 corners
```
**Solution:** Provide corners in format: `x1,y1 x2,y2 x3,y3 x4,y4`

### Performance Tips

1. **Start Small**: Test with a few images first
2. **Use Batch Mode**: More efficient for many images
3. **Resume Capability**: Can continue from where you left off
4. **Quality Control**: Review visualizations to ensure accuracy

## Integration with Training

Once annotations are complete, you can use them for training:

1. **Combine with Set 1**: Merge with existing annotations
2. **Train New Models**: Use the expanded dataset
3. **Validate Performance**: Test on both chess sets
4. **Domain Adaptation**: Ensure models work across different sets

## Support

If you encounter issues:

1. Check the logs for error messages
2. Verify the robust corner detection API is running
3. Ensure image paths are correct
4. Validate FEN format using chess notation
5. Review the visualization images for accuracy

## Next Steps

After completing annotation:

1. **Validate Annotations**: Review a sample of visualizations
2. **Combine Datasets**: Merge with existing Set 1 annotations
3. **Retrain Models**: Use expanded dataset for training
4. **Test Performance**: Validate on both chess sets
5. **Deploy Updates**: Update production models if improved

---

**Happy Annotating! 🎯♟️**
