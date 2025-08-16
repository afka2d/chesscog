# Quick Start Guide: Achieve 90% Piece Classification Accuracy

## 🎯 Goal: Improve from 48% to 90% accuracy

### Current Status
- **Current Accuracy**: 48.18%
- **Target Accuracy**: 90%
- **Improvement Needed**: 42 percentage points

## 🚀 Step-by-Step Implementation

### Phase 1: Dataset Expansion (Week 1)

#### Step 1.1: Generate Synthetic Data
```bash
cd /Users/tonyblum/code/chesscog
source venv/bin/activate
python expand_chess_dataset.py
```

**Expected Output:**
- Creates `expanded_dataset/` with 12,000 synthetic images (1,000 per class)
- Creates `enhanced_chess_dataset/` with merged original + synthetic data
- **Expected Improvement**: +15-20% accuracy

#### Step 1.2: Verify Dataset Expansion
```bash
# Check the new dataset size
find enhanced_chess_dataset -name "*.png" | wc -l
# Should show ~17,000+ images (original 5,631 + synthetic 12,000)
```

### Phase 2: Enhanced Model Training (Week 2)

#### Step 2.1: Train Enhanced Model
```bash
cd /Users/tonyblum/code/chesscog
source venv/bin/activate
python enhanced_piece_classifier.py
```

**Key Improvements:**
- **ResNet50** instead of ResNet18 (better feature extraction)
- **Advanced Data Augmentation** (perspective, noise, blur, color variations)
- **Focal Loss** (handles class imbalance)
- **AdamW Optimizer** with weight decay
- **Cosine Annealing** learning rate scheduling

**Expected Output:**
- Trains for 50 epochs
- Saves best model to `enhanced_models/best_model.pt`
- **Expected Improvement**: +20-25% accuracy (total: 65-70%)

### Phase 3: Model Ensemble (Week 3)

#### Step 3.1: Train Multiple Models
```python
# Train different architectures
models_to_train = [
    "ResNet50",
    "EfficientNet-B4", 
    "ConvNeXt",
    "Vision Transformer"
]

# Combine predictions using voting or weighted averaging
```

#### Step 3.2: Implement Ensemble
```python
# Create ensemble script
python ensemble_models.py
```

**Expected Improvement**: +10-15% accuracy (total: 75-85%)

### Phase 4: Fine-tuning & Optimization (Week 4)

#### Step 4.1: Hyperparameter Optimization
```python
# Use Optuna or similar for hyperparameter search
python optimize_hyperparameters.py
```

#### Step 4.2: Advanced Techniques
- **Test Time Augmentation (TTA)**
- **Multi-scale inference**
- **Post-processing rules**

**Expected Improvement**: +5-10% accuracy (total: 80-95%)

## 📊 Expected Results Timeline

| Week | Phase | Expected Accuracy | Cumulative Improvement |
|------|-------|-------------------|------------------------|
| 1 | Dataset Expansion | 60-65% | +12-17% |
| 2 | Enhanced Model | 70-75% | +22-27% |
| 3 | Ensemble | 80-85% | +32-37% |
| 4 | Optimization | 85-95% | +37-47% |

## 🛠️ Required Resources

### Hardware Requirements
- **GPU**: RTX 4090 or A100 (recommended)
- **RAM**: 32GB+ 
- **Storage**: 100GB+ for expanded dataset

### Software Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install matplotlib seaborn
pip install optuna  # for hyperparameter optimization
```

## 🎯 Success Metrics

### Primary Goals
- [ ] **Overall Accuracy > 90%**
- [ ] **Per-class accuracy > 85%** for all pieces
- [ ] **Queens & Kings accuracy > 80%** (currently 15-25%)

### Secondary Goals
- [ ] **Real-world performance validation**
- [ ] **Consistent results across different chess sets**
- [ ] **Fast inference time** (< 100ms per image)

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue 1: Out of Memory
```bash
# Reduce batch size
python enhanced_piece_classifier.py --batch_size 16
```

#### Issue 2: Overfitting
```python
# Increase dropout rate
dropout_rate = 0.5  # instead of 0.3
```

#### Issue 3: Poor Queen/King Performance
```python
# Increase synthetic data for these classes
target_samples_per_class = {
    'white_queen': 2000,
    'black_queen': 2000,
    'white_king': 2000,
    'black_king': 2000,
    # ... other pieces: 1000
}
```

## 📈 Monitoring Progress

### Daily Check-ins
1. **Dataset size verification**
2. **Training loss/accuracy curves**
3. **Validation accuracy tracking**
4. **Per-class performance analysis**

### Weekly Milestones
- **Week 1**: Dataset expanded to 17,000+ images
- **Week 2**: Enhanced model achieves 70%+ accuracy
- **Week 3**: Ensemble achieves 80%+ accuracy  
- **Week 4**: Final model achieves 90%+ accuracy

## 🎉 Success Criteria

### Minimum Viable Success
- **Overall accuracy > 85%**
- **All piece types > 70% accuracy**

### Target Success
- **Overall accuracy > 90%**
- **All piece types > 85% accuracy**
- **Queens & Kings > 80% accuracy**

### Stretch Goals
- **Overall accuracy > 95%**
- **Real-time performance** on mobile devices
- **Generalization** to different chess piece styles

## 🚀 Quick Commands

```bash
# 1. Expand dataset
python expand_chess_dataset.py

# 2. Train enhanced model
python enhanced_piece_classifier.py

# 3. Test accuracy
python test_uniform_model_accuracy.py

# 4. Deploy to production
# Update main.py to use enhanced model
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify hardware requirements
3. Review error logs
4. Consider reducing batch size or model complexity

---

**Remember**: The key to 90% accuracy is **data quality and quantity**. Focus on expanding the dataset first, then optimize the model architecture and training process. 