#!/usr/bin/env python3
"""
Chess Set Domain Analysis
=========================

Analyze the impact of training on multiple chess sets vs single chess set
and provide recommendations for generalization vs specialization.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessSetDomainAnalyzer:
    def __init__(self):
        self.data_dirs = [
            "grey_background_dataset/images",
            "my_chess_images"
        ]
        
    def analyze_visual_characteristics(self):
        """Analyze visual characteristics across different chess sets"""
        logger.info("🔍 ANALYZING CHESS SET VISUAL CHARACTERISTICS")
        logger.info("=" * 60)
        
        analysis_results = {
            'piece_styles': [],
            'board_materials': [],
            'lighting_conditions': [],
            'piece_colors': [],
            'size_variations': []
        }
        
        # Sample images from different datasets
        sample_images = []
        
        # Grey background dataset (main training set)
        grey_train_dir = Path("grey_background_dataset/images/train")
        if grey_train_dir.exists():
            grey_images = list(grey_train_dir.glob("*.JPG"))[:5]
            for img_path in grey_images:
                sample_images.append(("grey_background_main", img_path))
        
        # My chess images (potentially different set)
        my_images_dir = Path("my_chess_images/train/images")
        if my_images_dir.exists():
            my_images = list(my_images_dir.glob("*.JPG"))[:5]
            for img_path in my_images:
                sample_images.append(("my_chess_set", img_path))
        
        logger.info(f"📊 Found {len(sample_images)} sample images to analyze")
        
        # Analyze each image
        for dataset_name, img_path in sample_images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                # Analyze color characteristics
                mean_color = np.mean(img, axis=(0, 1))
                std_color = np.std(img, axis=(0, 1))
                
                # Analyze brightness
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                logger.info(f"   {dataset_name} - {img_path.name}:")
                logger.info(f"     Mean color (BGR): [{mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f}]")
                logger.info(f"     Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
                
                analysis_results['piece_colors'].append({
                    'dataset': dataset_name,
                    'image': img_path.name,
                    'mean_color': mean_color.tolist(),
                    'brightness': brightness,
                    'contrast': contrast
                })
                
            except Exception as e:
                logger.warning(f"   ⚠️  Could not analyze {img_path}: {e}")
        
        return analysis_results
    
    def analyze_training_implications(self):
        """Analyze the implications of single vs multi-set training"""
        logger.info("\n🧠 TRAINING STRATEGY ANALYSIS")
        logger.info("=" * 60)
        
        recommendations = {
            'single_set_pros': [
                "Higher accuracy on the specific set (specialized model)",
                "Faster training with less data needed",
                "More consistent performance on known conditions",
                "Simpler model with fewer parameters needed"
            ],
            'single_set_cons': [
                "Poor generalization to new chess sets",
                "Brittle to lighting/angle changes",
                "User frustration with different chess sets",
                "Limited commercial viability"
            ],
            'multi_set_pros': [
                "Better generalization across different chess sets",
                "More robust to lighting and environmental changes",
                "Better user experience (works with any chess set)",
                "Higher commercial value and wider market appeal",
                "More resilient to real-world variations"
            ],
            'multi_set_cons': [
                "Potentially lower accuracy on the original set",
                "Requires more diverse training data",
                "Longer training time and more complex models",
                "Risk of negative transfer if sets are too different"
            ]
        }
        
        logger.info("📈 SINGLE CHESS SET TRAINING:")
        for pro in recommendations['single_set_pros']:
            logger.info(f"   ✅ {pro}")
        for con in recommendations['single_set_cons']:
            logger.info(f"   ❌ {con}")
            
        logger.info("\n📈 MULTIPLE CHESS SET TRAINING:")
        for pro in recommendations['multi_set_pros']:
            logger.info(f"   ✅ {pro}")
        for con in recommendations['multi_set_cons']:
            logger.info(f"   ❌ {con}")
        
        return recommendations
    
    def provide_strategic_recommendations(self):
        """Provide strategic recommendations based on use case"""
        logger.info("\n🎯 STRATEGIC RECOMMENDATIONS")
        logger.info("=" * 60)
        
        strategies = {
            'general_model_approach': {
                'description': "Build one robust model for all chess sets",
                'implementation': [
                    "Collect training data from 5-10 different chess sets",
                    "Use aggressive data augmentation (lighting, angles, colors)",
                    "Implement domain adaptation techniques",
                    "Use larger, more robust model architectures (ResNet50+)",
                    "Apply regularization to prevent overfitting to any single set"
                ],
                'pros': [
                    "Best user experience - works with any chess set",
                    "Single model to maintain and deploy",
                    "Broader market appeal",
                    "More robust to real-world conditions"
                ],
                'cons': [
                    "May have slightly lower accuracy on your original set",
                    "Requires more diverse training data",
                    "More complex training process"
                ],
                'recommended_for': "Commercial applications, general use, App Store submission"
            },
            'chess_set_selection_approach': {
                'description': "Let users select their chess set type",
                'implementation': [
                    "Train separate models for different chess set types",
                    "Create a chess set detection/classification system",
                    "Implement model switching in the app",
                    "Maintain multiple specialized models"
                ],
                'pros': [
                    "Highest possible accuracy for each chess set",
                    "Can optimize for specific characteristics",
                    "Clear user control and expectations"
                ],
                'cons': [
                    "Complex app UI and user experience",
                    "Multiple models to train and maintain",
                    "Users may not know their chess set type",
                    "Increased app size and complexity"
                ],
                'recommended_for': "Professional/tournament use, chess enthusiasts"
            },
            'hybrid_approach': {
                'description': "General model with optional fine-tuning",
                'implementation': [
                    "Train a general model on diverse chess sets",
                    "Allow users to optionally 'calibrate' with 3-5 photos",
                    "Use few-shot learning to adapt to user's specific set",
                    "Fall back to general model if calibration fails"
                ],
                'pros': [
                    "Best of both worlds - general + specialized",
                    "Great user experience with optional optimization",
                    "Handles both casual and power users",
                    "Graceful degradation"
                ],
                'cons': [
                    "Most complex to implement",
                    "Requires advanced ML techniques",
                    "Additional UI complexity"
                ],
                'recommended_for': "Premium applications, advanced users"
            }
        }
        
        logger.info("🏆 RECOMMENDATION RANKING:")
        logger.info("1. 🥇 GENERAL MODEL APPROACH (Recommended for App Store)")
        logger.info("2. 🥈 HYBRID APPROACH (Best technical solution)")
        logger.info("3. 🥉 CHESS SET SELECTION (Niche use cases)")
        
        return strategies
    
    def analyze_current_dataset_diversity(self):
        """Analyze how diverse the current training data is"""
        logger.info("\n📊 CURRENT DATASET DIVERSITY ANALYSIS")
        logger.info("=" * 60)
        
        # Check annotation files for patterns
        annotation_files = list(Path("grey_background_dataset/annotations/train").glob("*.json"))
        
        image_characteristics = []
        
        for ann_file in annotation_files[:10]:  # Sample first 10
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Extract characteristics
                characteristics = {
                    'image_name': data.get('image_path', ann_file.stem),
                    'image_size': data.get('image_size', [0, 0]),
                    'has_fen': 'fen' in data and data['fen'] != "8/8/8/8/8/8/8/8 w - - 0 1",
                    'corners': data.get('corners', [])
                }
                
                image_characteristics.append(characteristics)
                
            except Exception as e:
                logger.warning(f"   Could not analyze {ann_file}: {e}")
        
        # Analyze patterns
        image_sizes = [char['image_size'] for char in image_characteristics if char['image_size'] != [0, 0]]
        unique_sizes = set(tuple(size) for size in image_sizes)
        
        logger.info(f"📸 Sample Analysis Results:")
        logger.info(f"   Images analyzed: {len(image_characteristics)}")
        logger.info(f"   Unique image sizes: {len(unique_sizes)}")
        logger.info(f"   Images with FEN: {sum(1 for c in image_characteristics if c['has_fen'])}")
        
        if len(unique_sizes) <= 2:
            logger.info("   🚨 LOW DIVERSITY: Most images appear to be from the same source/camera")
            return "low_diversity"
        else:
            logger.info("   ✅ GOOD DIVERSITY: Multiple image sources detected")
            return "good_diversity"

def main():
    analyzer = ChessSetDomainAnalyzer()
    
    print("🎯 CHESS SET DOMAIN ADAPTATION ANALYSIS")
    print("=" * 70)
    print("Analyzing the impact of training on multiple chess sets")
    print("vs specialized models for specific chess sets.")
    print()
    
    # Analyze visual characteristics
    visual_analysis = analyzer.analyze_visual_characteristics()
    
    # Analyze current dataset diversity
    diversity_level = analyzer.analyze_current_dataset_diversity()
    
    # Analyze training implications
    training_implications = analyzer.analyze_training_implications()
    
    # Provide strategic recommendations
    strategies = analyzer.provide_strategic_recommendations()
    
    print("\n" + "=" * 70)
    print("📋 EXECUTIVE SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    if diversity_level == "low_diversity":
        print("🚨 CURRENT STATE: Your training data appears to be from a single chess set")
        print("   This means your model is likely overfit to that specific set.")
        print()
        print("🎯 RECOMMENDED ACTION: BUILD A GENERAL MODEL")
        print("   Reason: Better user experience and broader market appeal")
        print()
        print("📋 IMPLEMENTATION PLAN:")
        print("   1. Collect images from 3-5 different chess sets")
        print("   2. Include various lighting conditions and angles")
        print("   3. Use aggressive data augmentation")
        print("   4. Accept 5-10% accuracy reduction on original set")
        print("   5. Gain 50-80% accuracy improvement on new sets")
        print()
        print("💡 ALTERNATIVE: If accuracy on your current set is critical,")
        print("   implement chess set selection in the app UI.")
    
    else:
        print("✅ CURRENT STATE: Good diversity detected in training data")
        print("   Your model should generalize reasonably well.")
        print()
        print("🎯 RECOMMENDED ACTION: ENHANCE CURRENT APPROACH")
        print("   Continue with general model but add more diverse data")
    
    print("\n🔗 NEXT STEPS:")
    print("   1. Test current model on images from a different chess set")
    print("   2. Measure accuracy drop to quantify the impact")
    print("   3. Decide based on actual performance degradation")
    print("   4. Consider user experience vs accuracy trade-offs")

if __name__ == "__main__":
    main()
