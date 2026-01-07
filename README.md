# MNIST Digit Classification

A comprehensive comparative study of classical machine learning algorithms for handwritten digit recognition using the MNIST dataset.

## 📋 Project Overview

This project implements and compares three classical machine learning algorithms for digit classification:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**

The analysis includes hyperparameter tuning, performance evaluation, error analysis, ensemble methods, and dimensionality reduction techniques using Principal Component Analysis (PCA).

### Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):
- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image dimensions**: 28×28 pixels
- **Pixel values**: 0-255 (normalized to 0-1)

## 🚀 Setup and Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**

The dataset will be automatically downloaded from Hugging Face when you run the notebook:
```python
!wget https://huggingface.co/datasets/Redfire-1234/mnist/resolve/main/mnist.zip -O mnist.zip
```

## 🏃 How to Run

### Option 1: Jupyter Notebook
```bash
jupyter notebook mnist_classification.ipynb
```

### Option 2: Google Colab
1. Upload the notebook to Google Colab
2. Run all cells sequentially (`Runtime` → `Run all`)

### Option 3: Python Script
```bash
python mnist_classification.py
```

## 🔑 Key Features

### 1. Data Preprocessing
- **Data Loading**: Automatic download and extraction from ZIP archive
- **Data Exploration**: Visual inspection of sample images and label distribution
- **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
- **Train-Test Split**: 80-20 split for validation
- **Dimensionality Reduction**: PCA with 50 components to reduce feature space from 784 to 50

### 2. Model Implementation

#### K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning**: Tested k values [3, 5, 7, 9]
- **Best Configuration**: Optimal k selected based on accuracy
- **Performance**: 97.3% accuracy (highest among all models)

#### Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Hyperparameter Tuning**: Grid search over C [0.1, 1, 10] and gamma ['scale', 0.01, 0.001]
- **Best Configuration**: Automatically selected based on validation accuracy
- **Performance**: 94.9% accuracy

#### Decision Tree
- **Hyperparameter Tuning**: Grid search over max_depth [5, 10, 15, 20] and min_samples_split [2, 5, 10]
- **Best Configuration**: Optimal parameters selected to balance complexity and performance
- **Performance**: 84.2% accuracy

### 3. Model Evaluation

#### Performance Metrics
- Accuracy scores for all models
- Confusion matrices with heatmap visualization
- Comparative bar charts

#### Error Analysis
- Visualization of misclassified digits
- Identification of common misclassification patterns
- Analysis of challenging digit pairs (e.g., 1 vs 7, 2 vs 7)

### 4. Advanced Techniques

#### Ensemble Learning
- **Voting Classifier**: Hard voting ensemble combining KNN, SVM, and Decision Tree
- **Performance**: 96.1% accuracy
- **Insight**: Demonstrates the trade-off between individual strong learners and ensemble robustness

#### PCA Impact Analysis
- Comparison of model performance with and without PCA
- **KNN**: Slight improvement with PCA (noise reduction)
- **SVM**: Significant drop with PCA (information loss)

## 📊 Results Summary

| Model | Accuracy (with PCA) | Accuracy (without PCA) |
|-------|---------------------|------------------------|
| **KNN** | **97.3%** | 97.4% |
| **SVM** | 94.9% | **98.2%** |
| **Decision Tree** | 84.2% | N/A |
| **Voting Ensemble** | 96.1% | N/A |

### Key Findings

1. **Best Individual Model**: KNN achieved the highest accuracy with PCA (97.3%)
2. **PCA Impact**: 
   - Beneficial for KNN (slight improvement)
   - Detrimental for SVM (significant drop from 98.2% to 94.9%)
3. **Common Misclassifications**: Digits 1, 2, and 7 frequently confused due to similar strokes
4. **Ensemble Performance**: Did not outperform best individual model due to weaker Decision Tree

## 🔍 Implementation Details

### Project Structure  

mnist-classification/  
├── mnist_classification.ipynb    # Main Jupyter notebook  
├── mnist_classification.py       # Python script version  
├── README.md                      # This file  
├── requirements.txt               # Python dependencies  



### Code Organization

The notebook is structured into 8 major sections:

1. **Data Acquisition and Loading**
2. **Exploratory Data Analysis**
3. **Data Preprocessing**
4. **Model Training and Hyperparameter Tuning**
5. **Model Evaluation**
6. **Key Findings**
7. **Ensemble Methods**
8. **PCA Impact Analysis**

### Reproducibility

- All random states are fixed (`random_state=42`)
- Complete code with all imports
- Step-by-step execution flow

## 💡 Insights and Recommendations

### Why KNN Performed Best (with PCA)
- Proximity-based classification works well for handwritten digits
- Similar digits cluster closely in reduced pixel space
- PCA helps by removing noise while preserving discriminative features

### Why SVM Suffered with PCA
- SVM relies on fine-grained decision boundaries
- PCA's dimensionality reduction removed critical information
- Full 784-dimensional space allows better separation

### Improvement Strategies

1. **Data Augmentation**: Rotation, shifting, and scaling to increase training diversity
2. **Feature Engineering**: Extract additional features like stroke patterns
3. **Deep Learning**: CNNs can automatically learn hierarchical features
4. **Optimal PCA Components**: Experiment with different numbers of components
5. **Ensemble Refinement**: Use models with similar performance levels

## 📚 References

- MNIST Dataset: [Yann LeCun's Website](http://yann.lecun.com/exdb/mnist/)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- PCA Analysis: [Principal Component Analysis Explained](https://en.wikipedia.org/wiki/Principal_component_analysis)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

Aman Ansari - [Your GitHub Profile](https://github.com/Redfire-1234)

## 🙏 Acknowledgments

- MNIST dataset creators and maintainers
- Scikit-learn community
- Hugging Face for dataset hosting
