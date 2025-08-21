#!/usr/bin/env python3
"""
Diagnostic script to understand why the piece classifier is performing poorly
and why it's predicting mostly knights for everything.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import json

def diagnose_model_issues():
    """Diagnose the piece classifier model issues."""
    
    print("🔍 Diagnosing Piece Classifier Model Issues")
    print("=" * 60)
    
    # Load the model
    model_path = Path("runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📁 Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = torch.load(str(model_path), map_location=device, weights_only=False)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Define transforms
    piece_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
        'black_queen', 'black_rook', 'white_bishop', 'white_king', 
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    # Test directory
    test_dir = Path("grey_background_dataset/pieces/test")
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"\n🔍 Analyzing model behavior on test dataset...")
    
    # Collect prediction statistics
    prediction_counts = defaultdict(int)
    confidence_scores = defaultdict(list)
    class_predictions = defaultdict(lambda: defaultdict(int))
    
    # Sample a few images from each class for detailed analysis
    samples_per_class = 5
    total_analyzed = 0
    
    for piece_class in piece_classes:
        class_dir = test_dir / piece_class
        if not class_dir.exists():
            continue
        
        test_images = list(class_dir.glob("*.png"))[:samples_per_class]
        print(f"\n🎯 Analyzing {piece_class}: {len(test_images)} samples")
        
        for img_path in test_images:
            try:
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_tensor = piece_transform(img).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = piece_classes[predicted_class_idx]
                
                # Update statistics
                prediction_counts[predicted_class] += 1
                confidence_scores[predicted_class].append(confidence)
                class_predictions[piece_class][predicted_class] += 1
                total_analyzed += 1
                
                # Show detailed prediction for first few
                if class_predictions[piece_class][predicted_class] <= 2:
                    status = "✅" if predicted_class == piece_class else "❌"
                    print(f"  {status} {img_path.name}: Predicted {predicted_class} (conf: {confidence:.3f})")
                
            except Exception as e:
                print(f"  ⚠️  Error processing {img_path.name}: {e}")
                continue
    
    print(f"\n📊 Analysis Results (Total images analyzed: {total_analyzed})")
    print("=" * 60)
    
    # Prediction distribution
    print(f"\n🎯 Prediction Distribution:")
    print("-" * 40)
    sorted_predictions = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
    for predicted_class, count in sorted_predictions:
        percentage = (count / total_analyzed) * 100
        print(f"  {predicted_class:15}: {count:3d} ({percentage:5.1f}%)")
    
    # Confidence analysis
    print(f"\n📈 Confidence Analysis:")
    print("-" * 40)
    for predicted_class, confidences in confidence_scores.items():
        if confidences:
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            print(f"  {predicted_class:15}: Avg {avg_conf:.3f}, Range [{min_conf:.3f}, {max_conf:.3f}]")
    
    # Confusion matrix (simplified)
    print(f"\n🔄 Confusion Matrix (Top 5 predictions per true class):")
    print("-" * 60)
    
    for true_class in piece_classes:
        if true_class in class_predictions:
            predictions = class_predictions[true_class]
            if predictions:
                # Sort by count
                sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  {true_class}:")
                for pred_class, count in sorted_preds[:5]:
                    percentage = (count / sum(predictions.values())) * 100
                    print(f"    → {pred_class:15}: {count:2d} ({percentage:5.1f}%)")
    
    # Model output analysis
    print(f"\n🧠 Model Output Analysis:")
    print("-" * 40)
    
    # Test with a single image to see raw outputs
    test_img_path = list(test_dir.glob("*/**/*.png"))[0]
    if test_img_path.exists():
        try:
            img = cv2.imread(str(test_img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = piece_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                raw_logits = output[0].cpu().numpy()
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            print(f"  Raw logits range: [{np.min(raw_logits):.3f}, {np.max(raw_logits):.3f}]")
            print(f"  Probabilities range: [{np.min(probabilities):.3f}, {np.max(probabilities):.3f}]")
            
            # Show top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            print(f"  Top 3 predictions:")
            for i, idx in enumerate(top_indices):
                print(f"    {i+1}. {piece_classes[idx]:15}: {probabilities[idx]:.3f}")
                
        except Exception as e:
            print(f"  ⚠️  Could not analyze model output: {e}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 40)
    
    if prediction_counts['black_knight'] > total_analyzed * 0.3:
        print("🚨 CRITICAL: Model is heavily biased toward predicting 'black_knight'")
        print("   - This suggests severe training data imbalance or model collapse")
        print("   - Consider retraining with balanced data or different architecture")
    
    if max(prediction_counts.values()) > total_analyzed * 0.5:
        print("⚠️  WARNING: Model is predicting one class for majority of inputs")
        print("   - This indicates the model has not learned meaningful features")
        print("   - Check training data quality and model architecture")
    
    # Check if confidence is artificially high
    avg_confidence = np.mean([np.mean(conf) for conf in confidence_scores.values() if conf])
    if avg_confidence > 0.8:
        print("⚠️  WARNING: Model shows artificially high confidence")
        print("   - This suggests overfitting or poor calibration")
        print("   - Consider regularization or ensemble methods")
    
    print(f"\n🔧 Next Steps:")
    print("   1. Check training data balance and quality")
    print("   2. Verify model architecture is appropriate")
    print("   3. Consider retraining with different hyperparameters")
    print("   4. Implement data augmentation and regularization")
    
    # Save detailed analysis
    analysis_results = {
        "total_analyzed": total_analyzed,
        "prediction_counts": dict(prediction_counts),
        "confidence_scores": {k: [float(x) for x in v] for k, v in confidence_scores.items()},
        "class_predictions": {k: dict(v) for k, v in class_predictions.items()},
        "analysis_timestamp": time.time()
    }
    
    with open("model_diagnosis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: model_diagnosis_results.json")

if __name__ == "__main__":
    import time
    diagnose_model_issues()
