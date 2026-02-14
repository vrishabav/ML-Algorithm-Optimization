# Classification

## 📌 Scope:

This repository contains the codebase used for implementations of the XGBoost method for Binary Classification, as outlined in the referenced paper - to predict on the MNIST data set whether a number being represented is even or odd

---

## 📁 Repository Structure

```
.
├── Even-Odd Classification: Report.pdf       # Full report with tabulations, summaries and thoughts
├── Even-Odd-XGBoost-README.md                # Overview and running guidelines
├── comparison.py                             # Comparison between Logistic Regression, Random Forest and XGBoost on the MNIST Dataset
├── hyperparam_tuning.py                      # Code used for tuning Hyperparameters of XGBoost based on the MNIST Dataset
└── xgboost.py                                # XGBoost implementation

```

---

## 📦 Installation & Dependencies

Python modules used: numpy, pandas

---

## ▶️ Running the Code

### Command-line

While running the code through command line, please make sure the datasets (MNIST_train, MNIST_val and MNIST_test) are in the same directory as the .py files
Kindly note that da24b033_xgboost.py and da24b033_comparison.py are already set to test on MNIST_test.csv, and the comparison results that were gotten from them were done using MNIST_validation.csv
Since the da24b033_hyperparam_tuning.py file has just been included for completeness, and its results are from the validation set, it has still been set to test on MNIST_validation.csv in case verification is required

---

## 🧾 Authors

Vrishab Anurag Venkataraghavan, DA24B033, IIT Madras (2025–26)
