[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras DA2401 Machine Learning Lab: End Semester Project

# MNIST Digit Classification (0-9)

---

## 📌 Scope (as per assignment brief)

This repository contains the codebase for my submission for the DA2401 End Semester Project, building a multi-class classification system to identify digits 0-9 from the MNIST dataset

The final system implemented in `main.py` is a **Voting Ensemble** that combines three distinct models:
1.  **K-Nearest Neighbors** (running on PCA-reduced data)
2.  **PCA Reconstruction Classifier** (anomaly-detection-based approach)
3.  **XGBoost Classifier** (One-vs-Rest configuration)

---

## 📁 Repository Structure

```
.

├── .github/                                    # Configuration

├── README.md                                   # Overview and running guidelines

├── report.md                                   # Full report with tabulations, summaries and thoughts

├── algorithms.py                               # Algorithm implementations for PCA, KNN, XGBoost, etc

├── hyperparam_tuning.py                        # Script used to tune hyperparameters for all the final models used (based on a subset of the data)

├── main.py                                     # Main script to train all models, build the ensemble, and evaluate

├── MNIST_train.csv                             # MNIST Training Dataset

└── MNIST_validation.csv                        # MNIST Validation Dataset
```

---

## 📦 Installation & Dependencies

* `numpy`
* `sklearn.metrics` (Specifically `f1_score`, `classification_report`, and `confusion_matrix`) for evaluation

---

## ▶️ Running the Code

### Command-line

1.  Ensure all `.py` files and the two `.csv` data files (`MNIST_train.csv`, `MNIST_validation.csv`) are in the same directory.
2.  Execute the `main.py` script to run the full training and evaluation pipeline:
    ```sh
    python main.py
    ```
3.  The script will load the data, train all three models, and print the performance (F1, train time, test time) of each individual model. Finally, it will print the F1 score, Classification Report, and Confusion Matrix for the final ensemble on the `MNIST_validation.csv` dataset. If testing is to be done on `MNIST_test.csv`, please change line 7 of `main.py`

If required, the `hyperparam_tuning.py` script can be run separately to reproduce the tuning results. It trains on a 20% subset of the data for speed

---

## 🧾 Author

Vrishab Anurag Venkataraghavan, DA24B033
