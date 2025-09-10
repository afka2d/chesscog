#!/usr/bin/env python3
"""
Test different piece classifier models to find the best one.
"""

import numpy as np
from PIL import Image
import chess
import torch
from pathlib import Path
from chesscog.recognition.recognition import ChessRecognizer

def test_model(model_path, model_name):
    """Test a specific model."""
    print(f"\n🧪 Testing {model_name}")
    print("=" * 40)
    
    try:
        # Load the model
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        
        # Load test image
        img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Test corners
        corners = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
        
        # Get occupancy
        recognizer = ChessRecognizer(Path("models"))
        board, detected_corners = recognizer.predict(img_array, chess.WHITE)
        
        occupancy = np.zeros(64, dtype=bool)
        for square in chess.SQUARES:
            if board.piece_at(square) is not None:
                occupancy[square] = True
        
        occupied_count = np.sum(occupancy)
        print(f"   Occupied squares: {occupied_count}/64")
        
        # Test piece classification (simplified version)
        from chesscog.core.dataset import build_transforms, Datasets, name_to_piece
        from recap import CfgNode as CN
        
        # Load config
        cfg_path = model_path.replace('.pt', '.yaml')
        if Path(cfg_path).exists():
            cfg = CN.load_yaml_with_base(cfg_path)
            transforms = build_transforms(cfg, mode=Datasets.TEST)
            piece_classes = np.array(list(map(name_to_piece, cfg.DATASET.CLASSES)))
        else:
            print(f"   ⚠️  No config file found for {model_name}")
            return None
        
        # Extract pieces from occupied squares
        piece_imgs = []
        for i, is_occupied in enumerate(occupancy):
            if is_occupied:
                rank, file = i // 8, i % 8
                # Extract piece image (simplified)
                piece_imgs.append(img)  # This is simplified - in reality we'd extract the actual piece
        
        if len(piece_imgs) > 0:
            # Apply transforms
            piece_imgs = [transforms(img) for img in piece_imgs]
            piece_imgs = torch.stack(piece_imgs)
            
            # Get predictions
            with torch.no_grad():
                predictions = model(piece_imgs)
                predicted_classes = predictions.argmax(axis=-1).cpu().numpy()
                piece_names = piece_classes[predicted_classes]
            
            # Analyze results
            piece_names_str = []
            for piece in piece_names:
                if hasattr(piece, 'symbol'):
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    piece_names_str.append(piece_name)
                else:
                    piece_names_str.append(str(piece))
            
            unique_types = set(piece_names_str)
            pawn_count = sum(1 for name in piece_names_str if 'p' in name.lower())
            
            print(f"   Pieces detected: {len(piece_names_str)}")
            print(f"   Unique types: {len(unique_types)}")
            print(f"   Types: {list(unique_types)}")
            print(f"   Pawn count: {pawn_count}/{len(piece_names_str)}")
            print(f"   Pawn ratio: {pawn_count/len(piece_names_str):.2f}")
            
            return {
                'model_name': model_name,
                'pieces': len(piece_names_str),
                'unique_types': len(unique_types),
                'pawn_ratio': pawn_count/len(piece_names_str),
                'types': list(unique_types)
            }
        else:
            print(f"   ⚠️  No occupied squares found")
            return None
            
    except Exception as e:
        print(f"   ❌ Error testing {model_name}: {e}")
        return None

def main():
    """Test all available models."""
    print("🔍 Testing Different Piece Classifier Models")
    print("=" * 60)
    
    models_dir = Path("models/piece_classifier")
    models_to_test = [
        ("InceptionV3.pt", "InceptionV3 (current)"),
        ("ResNet_robust.pt", "ResNet Robust"),
        ("ResNet_robust_full.pt", "ResNet Robust Full"),
        ("ResNet_simple_robust.pt", "ResNet Simple Robust"),
        ("ResNet_simple_balanced.pt", "ResNet Simple Balanced"),
        ("ResNet_simple.pt", "ResNet Simple"),
        ("ResNet_lightweight.pt", "ResNet Lightweight")
    ]
    
    results = []
    
    for model_file, model_name in models_to_test:
        model_path = models_dir / model_file
        if model_path.exists():
            result = test_model(model_path, model_name)
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  Model {model_name} not found at {model_path}")
    
    # Compare results
    print(f"\n📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Pieces':<8} {'Types':<6} {'Pawn%':<8} {'Types'}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<25} {result['pieces']:<8} {result['unique_types']:<6} {result['pawn_ratio']*100:<7.1f}% {result['types']}")
    
    # Find best model
    if results:
        best_model = min(results, key=lambda x: x['pawn_ratio'])
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   Pawn ratio: {best_model['pawn_ratio']*100:.1f}%")
        print(f"   Unique types: {best_model['unique_types']}")
        print(f"   Types detected: {best_model['types']}")

if __name__ == "__main__":
    main()
