# Chess Recognition Model Training Guide

This guide will help you train the chess recognition models with your own chess board images to improve accuracy.

## Overview

The chess recognition system uses two neural networks:
1. **Occupancy Classifier** - Determines which squares have pieces
2. **Piece Classifier** - Identifies what type of piece is on occupied squares

## Step 1: Prepare Your Images

1. **Collect chess board images** with different:
   - Lighting conditions
   - Board types (vinyl, wooden, tournament)
   - Camera angles
   - Piece arrangements

2. **Create a directory structure**:
   ```
   my_chess_images/
   ├── train/
   │   ├── images/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── annotations/
   ├── val/
   │   ├── images/
   │   └── annotations/
   └── test/
       ├── images/
       └── annotations/
   ```

## Step 2: Create Annotation Templates

Run the dataset creation script to generate annotation templates:

```bash
python create_custom_dataset.py \
  --input_dir my_chess_images/train/images \
  --output_dir my_chess_images/train \
  --create_templates
```

This will create JSON files in `my_chess_images/train/annotations/` that you need to edit.

## Step 3: Annotate Your Images

For each image, you need to annotate:

### A. Corner Coordinates
The 4 corners of the chess board in pixel coordinates:
- **Top-left corner** (a8 square)
- **Top-right corner** (h8 square) 
- **Bottom-right corner** (h1 square)
- **Bottom-left corner** (a1 square)

### B. FEN Notation
The chess position in FEN format. For example:
- Empty board: `8/8/8/8/8/8/8/8 w - - 0 1`
- Starting position: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
- Your position: `8/8/8/4P3/3p4/8/8/8 w - - 0 1`

### Example Annotation
```json
{
  "image_path": "IMG_4698.JPG",
  "image_size": [3240, 5760],
  "corners": [
    [500, 800],    # Top-left (a8)
    [2740, 800],   # Top-right (h8)
    [2740, 4960],  # Bottom-right (h1)
    [500, 4960]    # Bottom-left (a1)
  ],
  "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
  "white_turn": true,
  "notes": "White pawn on e5, black pawn on d4"
}
```

## Step 4: Validate Annotations

Check that all annotations are correct:

```bash
python create_custom_dataset.py \
  --input_dir my_chess_images \
  --validate
```

## Step 5: Train the Models

Train both occupancy and piece classifiers:

```bash
python train_custom_models.py \
  --custom_data_dir my_chess_images \
  --output_dir trained_models
```

This will:
1. Prepare your dataset in the expected format
2. Train the occupancy classifier
3. Train the piece classifier
4. Save the trained models

## Step 6: Use Your Trained Models

Update the model paths in your configuration files to use the new trained models:

```yaml
# config/occupancy_classifier/ResNet.yaml
MODEL:
  PATH: "trained_models/occupancy_classifier/model/best.pth"

# config/piece_classifier/ResNet.yaml  
MODEL:
  PATH: "trained_models/piece_classifier/model/best.pth"
```

## Tips for Better Training

### 1. Diverse Data
- Include images with different lighting (bright, dim, shadows)
- Different board materials (vinyl, wood, plastic)
- Various camera angles (slight perspective is good)
- Different piece sets (Staunton, tournament, etc.)

### 2. Annotation Quality
- Be precise with corner coordinates
- Double-check FEN notation
- Include both simple and complex positions
- Balance between empty squares and occupied squares

### 3. Dataset Size
- **Minimum**: 50-100 images per subset (train/val/test)
- **Recommended**: 200+ images per subset
- **Optimal**: 500+ images per subset

### 4. Training Time
- Small dataset (50-100 images): 10-30 minutes
- Medium dataset (200-500 images): 1-3 hours
- Large dataset (500+ images): 3-8 hours

## Troubleshooting

### Common Issues

1. **"Corners appear to be default values"**
   - You haven't updated the corner coordinates in the JSON files
   - Use the annotated images to help identify corners

2. **"Invalid FEN"**
   - Check your FEN notation with a chess validator
   - Make sure the position is legal

3. **Training fails**
   - Ensure you have enough images in each subset
   - Check that all image files exist and are readable
   - Verify annotation format is correct

### Getting Help with Corner Detection

Use the annotation tool to create visual guides:

```bash
python create_custom_dataset.py \
  --input_dir my_chess_images \
  --create_annotated_images
```

This creates images with numbered corner points to help you identify the correct coordinates.

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Create dataset structure
mkdir -p my_chess_images/{train,val,test}/{images,annotations}

# 2. Copy your images
cp IMG_4698.JPG my_chess_images/train/images/
cp sample.jpeg my_chess_images/train/images/

# 3. Create annotation templates
python create_custom_dataset.py \
  --input_dir my_chess_images/train/images \
  --output_dir my_chess_images/train \
  --create_templates

# 4. Edit the JSON files in my_chess_images/train/annotations/
# Add corner coordinates and FEN notation

# 5. Validate annotations
python create_custom_dataset.py \
  --input_dir my_chess_images \
  --validate

# 6. Train models
python train_custom_models.py \
  --custom_data_dir my_chess_images \
  --output_dir trained_models
```

## Next Steps

After training, test your new models:

1. Update the model paths in your configuration
2. Restart the API server
3. Test with your images to see improved accuracy

The more diverse and accurately annotated your training data, the better the recognition performance will be! 