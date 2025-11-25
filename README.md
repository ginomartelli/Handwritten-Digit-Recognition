# Handwritten Digit Recognition

A machine learning project that classifies digits from 8x8 pixel images using feature engineering and multiple models.

- **Feature Engineering**: Combined PCA (20 components), zonal partitioning (3 regions), and Sobel edge detection
- **Models**: Optimized SVC with linear/RBF kernels and Neural Networks
- **Data**: 1797 handwritten digit images from scikit-learn

## Results

| Model | Test Accuracy |
|-------|---------------|
| SVC (Linear) | **99.3%** |
| Neural Network | 95.8% |

## Key Findings

- **Feature engineering** reduced dimensions from 64 to 24 while improving performance
- **SVC outperformed** neural networks for this small dataset
- **Proper preprocessing** eliminated overfitting in neural networks

The project successfully demonstrates that machines can learn to recognize handwritten digits with high accuracy (99.3%) using optimized feature extraction and traditional ML models.
