# Machine Learning From Scratch

## Overview
This repository contains mathematical implementations of fundamental Machine Learning algorithms from scratch using Python and NumPy. The goal is to demonstrate the core mechanics, optimization techniques, and mathematical foundations underlying these models without relying on high-level abstractions like scikit-learn.

## Algorithms Implemented


    
### Supervised Learning

| Algorithm | Type | Key Concepts |
|-----------|------|--------------|
| **[Linear Regression](./Linear%20Regression)** | Regression | Gradient Descent, Mean Squared Error (MSE) |
| **[Logistic Regression](./Logistic%20Regression)** | Classification | Sigmoid Function, Log-Loss, Binary Classification |
| **[Decision Tree](./DecisionTree)** | Classification | Entropy, Gini Impurity, Information Gain |
| **[Random Forest](./Random%20Forest)** | Ensemble | Bagging, Feature Randomness, Voting Mechanics |
| **[Naive Bayes](./Naive%20Bayes)** | Classification | Bayes' Theorem, Gaussian Distribution |
| **[KNN](./KNN)** | Reg/Class | Euclidean Distance, K-Nearest Neighbors Voting |
| **[SVM](./SVM)** | Classification | Hinge Loss, Hyperplane Optimization |
| **[Perceptron](./Perceptron)** | Classification | Single-Layer Neural Network Logic |
| **[AdaBoost](./AdaBoost)** | Ensemble | Weak Classifiers, Sample Weighting |

### Unsupervised Learning

| Algorithm | Type | Key Concepts |
|-----------|------|--------------|
| **[PCA](./PCA)** | Dim. Reduction | Covariance Matrix, Eigendecomposition |
| **[LDA](./LDA)** | Dim. Reduction | Fisher's Linear Discriminant, Class Separation |
| **[K-Means](./K-Means%20Clustering)** | Clustering | Centroid Initialization, Euclidean Distance |

## Prerequisites
* Python 3.x
* NumPy
* Matplotlib (for visualizations)

## Usage
Each algorithm is contained within its own directory. You can run the demonstration script inside each module to see the training process and results.

**Example:**

```bash
cd "Linear Regression"
python main.py

