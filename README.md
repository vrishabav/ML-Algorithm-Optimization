# ML-Algorithm-Optimization

**Author:** Vrishab Anurag Venkataraghavan

---

## Overview

This repository contains implementations and analyses of various Machine Learning optimization techniques and architectures. The projects range from comparing first-order vs. second-order optimization methods for Linear Regression to building custom Gradient Boosting frameworks and high-performance Voting Ensembles for digit classification.

---

## Repository Structure

```
ML-Algorithm-Optimization/
├── Linear-Regression-Variations/     # Optimization methods for Regression (SGD, Adam, SSHF, etc.)
├── MNIST-even-odd-classification/    # Custom XGBoost implementation for Binary Classification
├── MNIST-digit-classification/       # Voting Ensemble for Multi-class Classification
├── MNIST_train.csv                   # Training dataset for classification tasks
├── MNIST_validation.csv              # Validation dataset for classification tasks
└── README.md
```

---

# Project 1: Linear Regression Variations

**Location:** `/Linear-Regression-Variations`

## Description

This module explores different optimization algorithms applied to Linear Regression using the `diamonds.csv` dataset. The goal was to compare convergence rates, stability, and final Mean Squared Error (MSE) between first-order and higher-order methods.

---

## Methods Implemented

1. **Stochastic Gradient Descent (SGD)** - Baseline first-order method  
2. **Adam (Adaptive Moment Estimation)** - First-order method with adaptive learning rates and momentum  
3. **ADMM (Alternating Direction Method of Multipliers)** - Decomposes the problem into sub-problems (Ridge Regression + Quadratic Minimization)  
4. **Stochastic Quasi-Newton (SQN)** - Uses L-BFGS recursion to approximate the inverse Hessian  
5. **Sub-sampled Hessian-Free (SSHF)** - Second-order method using Conjugate Gradient (CG) to solve the Newton system via Hessian-vector products  

---

## Key Results

The **Sub-sampled Hessian-Free (SSHF)** method was the unanimous winner, demonstrating the superior accuracy of second-order information.

| Algorithm | Type | Test MSE | Performance Note |
|------------|------------|------------|------------------|
| **SGD** | First-Order | 2.69 × 10¹² | Highly unstable; diverged without extensive tuning |
| **Adam** | First-Order (Adaptive) | 2.99 × 10⁷ | Stable; converged quickly |
| **ADMM** | First-Order (Splitting) | 2.80 × 10⁷ | Slightly outperformed Adam |
| **SQN** | Quasi-Second-Order | 2.99 × 10⁷ | Comparable to Adam despite curvature approximation |
| **SSHF** | Second-Order | **2.06 × 10⁷** | **Best Performer. Lowest MSE across all sets** |

---

# Project 2: MNIST Even-Odd Classification

**Location:** `/MNIST-even-odd-classification`

## Description

A binary classification task to determine whether a handwritten digit is Even or Odd. The core of this project is a **custom implementation of the XGBoost Classifier**, tuned via grid search and compared against Logistic Regression and Random Forest.

---

## Custom XGBoost Implementation

The implementation features:

- **Exact Greedy & Approximate Algorithms** for split finding  
- **Sparsity Handling** optimized for the sparse nature of MNIST data  
- **Hyperparameter Tuning** via grid search over 32 combinations  

---

## Performance Comparison

The XGBoost model significantly outperformed traditional classifiers in accuracy, though at a higher computational cost.

| Model | Accuracy | Training Time (s) | Precision | Recall | F1 Score |
|--------|------------|------------------|------------|------------|------------|
| **Logistic Regression** | 0.8792 | **10.27** | 0.8801 | 0.8730 | 0.8765 |
| **Random Forest** | 0.9088 | 108.65 | 0.9433 | 0.8644 | 0.9032 |
| **Custom XGBoost** | **0.9632** | 158.17 | **0.9595** | **0.9658** | **0.9627** |

### Optimal Hyperparameters

- `max_depth = 6`  
- `subsample = 0.8`  
- `min_child_weight = 5`  
- `gamma = 0`  
- `reg_lambda = 1.0`  

---

# Project 3: MNIST Digit Classification

**Location:** `/MNIST-digit-classification`

## Description

The final system architecture is a **Voting Ensemble** designed to classify all 10 MNIST digits. The system was optimized to balance high accuracy with a strict training/inference time constraint (under 5 minutes total runtime).

---

## Architecture

The ensemble averages probability distributions from three distinct models to maximize robustness:

### 1. K-Nearest Neighbors (KNN)
- Input: PCA-reduced data (30 components)  
- Role: Fast, distance-based classification  

### 2. PCA Reconstruction Classifier (Anomaly Detection)
- Trains a separate PCA model (50 components) for each digit class  
- Predicts the class with the lowest reconstruction error  
- Role: Unsupervised feature correlation  

### 3. XGBoost (One-vs-Rest)
- 10 binary classifiers trained on raw data  
- Config:
  - `n_estimators = 40`
  - `max_depth = 5`
  - `learning_rate = 0.2`
  - `colsample_bytree = 0.5`
- Role: Strong gradient-boosted decision boundaries  

---

## Final Results

The ensemble strategy proved robust, correcting the individual errors of component models.

- **Total Runtime:** 254.64s (within the 300s limit)  
- **Ensemble Weighted F1 Score:** **0.9691**

| Model Component | Individual F1 Score | Train Time (s) | Predict Time (s) |
|------------------|----------------------|----------------|------------------|
| **XGBoost (OvR)** | 0.9676 | 243.19 | 2.08 |
| **KNN (PCA)** | 0.9559 | ~0.00 | 2.97 |
| **PCA Anomaly** | 0.9383 | 1.05 | 0.16 |

---

## Usage

### Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib xgboost
```

### Running the Final Digit Classification Project

```bash
cd MNIST-digit-classification
python main.py
```

---

## Final Summary

- **Best Accuracy:** SSHF (Linear Regression)  
- **Best Binary Classifier:** Custom XGBoost  
- **Best Overall System:** Voting Ensemble (Weighted F1 = 0.9691)  
- **Best Trade-off (Speed vs Performance):** Adam / ADMM (Regression), XGBoost (Classification)  
