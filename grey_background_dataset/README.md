# Grey Background Chess Dataset

This dataset is specifically designed for training chess recognition models on images with grey backgrounds.

## Directory Structure

```
grey_background_dataset/
├── images/           # Place your grey background chess images here
├── annotations/      # JSON annotation files (auto-generated)
├── processed/        # Processed training data
└── models/          # Trained models
```

## Setup Instructions

1. **Add Images**: Place your grey background chess images in the `images/` directory
   - Supported formats: JPG, JPEG, PNG
   - Images should have grey backgrounds for consistent training

2. **Create Annotations**: Run the annotation creation script:
   ```bash
   python create_custom_dataset.py --input_dir grey_background_dataset/images --output_dir grey_background_dataset
   ```

3. **Update Corner Coordinates**: Use the corner update tool:
   ```bash
   python update_corners.py --interactive
   ```

4. **Train Models**: Run the training script:
   ```bash
   python enhanced_batch_train.py --full_pipeline
   ```

## Image Requirements

- **Background**: Grey background (consistent lighting)
- **Chess Board**: Clear 8x8 grid visible
- **Pieces**: Standard chess pieces clearly visible
- **Quality**: Good resolution, well-lit, no blur

## Expected Results

Training on grey background images should improve recognition accuracy for:
- Images with similar grey backgrounds
- Consistent lighting conditions
- Standard chess board setups

## Notes

- This dataset is optimized for grey background scenarios
- Corner coordinates are crucial for accurate training
- FEN notation should be provided for each position
