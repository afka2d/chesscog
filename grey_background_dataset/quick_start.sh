#!/bin/bash
# Quick Start Script for Grey Background Chess Dataset

echo "🎯 Setting up Grey Background Chess Dataset"

# Step 1: Check if images are present
if [ ! "$(ls -A grey_background_dataset/images)" ]; then
    echo "❌ No images found in grey_background_dataset/images/"
    echo "Please add your grey background chess images to grey_background_dataset/images/"
    exit 1
fi

echo "✅ Found images in grey_background_dataset/images/"

# Step 2: Create annotations
echo "📝 Creating annotations..."
python create_custom_dataset.py --input_dir grey_background_dataset/images --output_dir grey_background_dataset

# Step 3: Show next steps
echo ""
echo "🎯 Next Steps:"
echo "1. Update corner coordinates: python update_corners.py --interactive"
echo "2. Add FEN notations to annotation files"
echo "3. Train the model: python enhanced_batch_train.py --full_pipeline"
echo ""
echo "📖 See grey_background_dataset/README.md for detailed instructions"
