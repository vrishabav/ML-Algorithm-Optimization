# Digit Classification Report

Vrishab Anurag Venkataraghavan

---

## 1. Models and Architecture

The final system architecture in `main.py`, is a **Voting Ensemble** between 3 models:

1.  **K-Nearest Neighbors (KNN):** Simple KNN which takes in PCA-reduced input
2.  **PCA Reconstruction Classifier (PCA Anomaly):** Anomaly-detection approach, which trains a separate PCA model for each digit, and calculates the reconstruction error against all 10 class-specific PCA models for any new image. It predicts the class with lowest reconstruction error
3.  **XGBoost (One-vs-Rest):** As implemented in Assignment 2, but modified to be OneVsRest for 10-class classification. Again done on the raw non-PCA data (as it was found to perform ~4% worse on PCA data)


**Ensemble Prediction:** A `VotingClassifier` as defined in `algorithms.py` takes the three trained models as estimators. When `predict_proba` is called, it gathers the probability distributions from all three models (using PCA data for KNN and raw data for the others) and averages them. The final prediction is the class with the highest average probability.

**Evaluation:** The system evaluates the performance of each individual model and the final ensemble on the validation dataset using the **weighted F1 score**.

---

## 2. Hyper-parameter Tuning

Hyper-parameters were tuned using the `hyperparam_tuning.py` script. To ensure a fast tuning process, models were trained on a 20% subset (2000 samples) of the training data and evaluated against the full validation set (2499 samples)

### 2.1) PCA + KNN Classifier

A grid search was performed to find the optimal number of PCA components and the optimal k for KNN.

* **PCA `n_components` Tested:** [30, 40, 50]
* **KNN `k` Tested:** [3, 5, 7]

| PCA Components | k | Weighted F1 Score |
| :--- | :-- | :--- |
| 30 | 3 | **0.9333** |
| 30 | 5 | 0.9322 |
| 30 | 7 | 0.9221 |
| 40 | 3 | 0.9329 |
| 40 | 5 | 0.9242 |
| 40 | 7 | 0.9185 |
| 50 | 3 | 0.9293 |
| 50 | 5 | 0.9237 |
| 50 | 7 | 0.9157 |

Best combination was **`pca_n_components = 30`** and **`knn_k = 3`**, achieving an F1 score of 0.9333

### 2.2) PCA Reconstruction Classifier

The number of components for each per-class PCA model was tuned.

* **`n_components` Tested:** [20, 30, 40, 50]

| `n_components` | Weighted F1 Score |
| :--- | :--- |
| 20 | 0.9351 |
| 30 | 0.9376 |
| 40 | 0.9378 |
| 50 | **0.9379** |

Best **`n_components`** was gotten to be 50

### 2.3) XGBoost Classifier (One-vs-Rest)

A grid search was conducted on `n_estimators`, `max_depth`, and `learning_rate`

* **`n_estimators` Tested:** [30, 40]
* **`max_depth` Tested:** [4, 5]
* **`learning_rate` Tested:** [0.2, 0.3]

| `n_estimators` | `max_depth` | `learning_rate` | Weighted F1 Score |
| :--- | :--- | :--- | :--- |
| 30 | 4 | 0.2 | 0.9379 |
| 30 | 4 | 0.3 | 0.9327 |
| 30 | 5 | 0.2 | 0.9362 |
| 30 | 5 | 0.3 | 0.9382 |
| 40 | 4 | 0.2 | 0.9350 |
| 40 | 4 | 0.3 | 0.9411 |
| 40 | 5 | 0.2 | **0.9455** |
| 40 | 5 | 0.3 | 0.9418 |

Best parameters were **`n_estimators = 40`**, **`max_depth = 5`**, and **`learning_rate = 0.2`**, achieving the highest individual F1 score of 0.9455

### Final Parameters

`main.py` script uses these parameters:
* **Global PCA:** `n_components = 30`
* **KNN:** `k = 3`
* **PCA Reconstruction:** `n_components = 50`
* **XGBoost:**
    * `n_estimators = 40`
    * `learning_rate = 0.2`
    * `max_depth = 5`
    * `colsample_bytree = 0.5`

---

## 3. Optimization and Evaluation

### Optimizations

1.  **Dimensionality Reduction with PCA:** The K-Nearest Neighbors algorithm's prediction time is heavily dependent on the number of features. Running it on all 784 features is computationally infeasible. By using PCA to reduce the data to **30 components**, the prediction time is dramatically reduced (to around **5.28 seconds**). When checking the full data for both train and val on KNN, *without* implementing PCA (training with all 784 features) the weighted F1 for val was gotten to be 0.9477 which is even worse than what we got with `PCA30+KNN`; while taking more than 100x the testing time

2.  **Model Selection & Evolution:** Initial trials explored a wider range of models, including `RandomForestClassifier` and `AdaBoostSAMMEClassifier`.
    * `RandomForest` proved to be extremely slow, with one trial taking **133.42 seconds** to train, with no significantly good accuracy either
    * `AdaBoost` was both slow (**30.43 seconds**) and had very poor accuracy (F1 = 0.7870)
    * These slow and inefficient models were removed from the final architecture. They were replaced by the `XGBoostClassifier` from assignment 2, modified to do OneVsRest classification. Its performance was observed to be much superior

4.  **Efficient XGBoost Implementation (from Assignment 2):** The custom `XGBoostTree` in `algorithms.py` includes optimizations to speed up training:
    * **Sparsity Handling:** It explicitly handles feature values of 0 (which show up a lot in MNIST) to avoid unnecessary split calculations
    * **Quantile-Based Splitting:** It uses a quantile-based sketching approach to find candidate split points, rather than iterating over every unique feature value, which is crucial for continuous-valued features

### Evaluation Results

The final models were trained on the full `MNIST_train.csv` dataset (10,002 samples) and evaluated on the `MNIST_validation.csv` dataset (2499 samples). The total training and evaluation runtime was **254.64 seconds**, which is successfully within the 5-minute requirement

**Final Model Performance (on Full Training Set)**
| Model | Weighted F1 | Train Time (s) | Predict Time (s) |
| :--- | :--- | :--- | :--- |
| **XGB** | 0.9676 | 243.19 | 2.08 |
| **KNN** | 0.9559 | 0.00 | 2.97 |
| **PCA_Anomaly** | 0.9383 | 1.05 | 0.16 |

The `XGBoost` model emerged as the single best-performing model, outperforming its tuning F1 score (0.9455). `KNN` remained a strong and exceptionally fast (0.00s train time) model. `PCA_Anomaly` was the fastest to predict (0.16s)

**Final Ensemble Performance**
The `VotingClassifier` ensemble, which averages the probabilities of these three models, achieved the highest performanc:

* **Ensemble Weighted F1 Score:** **0.9691**
* **Ensemble Prediction Time:** 4.98s
* **Total System Runtime:** 254.64s

**Ensemble Classification Report**
|   | precision | recall | f1-score | support |
|---|-----------|--------|----------|---------|
| 0 | 0.97      | 1.00   | 0.98     | 247     |
| 1 | 0.97      | 0.99   | 0.98     | 281     |
| 2 | 0.97      | 0.96   | 0.97     | 248     |
| 3 | 0.97      | 0.94   | 0.96     | 255     |
| 4 | 0.97      | 0.98   | 0.97     | 243     |
| 5 | 0.97      | 0.97   | 0.97     | 226     |
| 6 | 0.98      | 0.98   | 0.98     | 246     |
| 7 | 0.95      | 0.98   | 0.97     | 261     |
| 8 | 0.99      | 0.95   | 0.97     | 244     |
| 9 | 0.95      | 0.94   | 0.95     | 248     |
| **accuracy** |   |  | **0.97** | **2499** |
| **macro avg** | 0.97 | 0.97 | 0.97 | 2499 |
| **weighted avg** | 0.97 | 0.97 | 0.97 | 2499 |

---

**Ensemble Confusion Matrix**

```
 [246   0   0   0   0   0   1   0   0   0]
 [  0 278   1   1   0   0   0   0   0   1]
 [  1   0 239   0   3   0   1   2   2   0]
 [  0   2   2 240   0   3   0   5   1   2]
 [  1   0   0   0   0 238   0   0   0   4]
 [  0   2   1   3   0 219   1   0   0   0]
 [  4   0   0   0   0   1 241   0   0   0]
 [  0   2   1   0   0   0   0 257   0   1]
 [  2   1   2   0   0   2   2   0 232   3]
 [  0   1   0   3   5   1   0   6   0 232]
```

---

## 4. Summary of Thoughts & Observations

This project made it interesting to combine different things we've learned through classes and the other assignments, and ultimately apply it to a real world problem (albeit one that has been solved many times over the years). A few of the observations and ideas throughout this process include:

* **Model Evolution is Important:** The initial "kitchen sink" approach, which included `RandomForest` and `AdaBoost`, was slow and ineffective (and also  less accurate). `RandomForest` took 133 seconds for a 0.90 F1, while `AdaBoost` took 30 seconds for a disastrous 0.78 F1. Removing these and replacing them with a single, highly-optimized `XGBoost` model (which achieved a 0.9455 F1 in tuning) was an important decision

* **Diversity in Ensembles:** The final ensemble is strong because it combines three fundamentally different approaches:
    1.  **KNN:** Local neighborhood similarity
    2.  **PCA Anomaly (unsupervised, based on reconstructions):** Class-wise feature correlation
    3.  **XGBoost (Gradient-based and decision tree framework):** Strong, boosted decision boundary

This diversity makes the ensemble robust, as the models are likely to make different types of errors, which can be corrected by the vote. This makes the ensemble greater than the sum of its parts, and a very effective classification model achieving ~97% accuracy

* **Bottlenecks:** KNN's prediction time and Random Forest's training time were the two biggest performance hurdles. The solutions (PCA for KNN, and replacing RF with XGB) were critical to making the model execute in a reasonable amount of time (well within the 5-minute time limit). Throughout the different iterations, KNN and PCA Anomaly were always the top performers, justifying their inclusion in the final ensemble

* **Bias-Variance Analysis:** Using a relatively low `k=3` for the KNN makes this theoretically a high-variance, low-bias model. However, no significant change in performance was seen for various greater values of k. To keep the time complexity lower, and since the variance was anyway being managed by the ensemble, `k=3` was kept. XGBoost is a low-bias, high-variance model by nature; while PCA Anomaly is high-bias, low-variance (it assumes each digit is best represented by its own class' 50-dimensional subspace). Therefore the effects tend to cancel out and reduce the overall variance when the ensemble is applied and probabilities are averaged

* **Common Model Confusions:** An examination of the final ensemble's confusion matrix reveals specific, challenging digit pairs, which points to the intrinsic ambiguity of the dataset, and the challenges that come with digit classification
    * The most common error was confusing a 9 for a 7, featuring in six instances
    * Symmetrical confusion was also relatively common; 4s were confused with 9s four times, and 9s were confused with 4s five times. Intuitively this makes sense as a "closed" 4 is very similar to a 9
    * Another symmetric pair was 3 and 5. The model misclassified three 3s as 5s, and three 5s as 3s
    * The digit 8 was also difficult, with its 232 correct predictions offset by errors across several other classes. This shows that its loops can be misread in many different ways. These error patterns do not just reflect a flaw in the models' robustness, but also the inherent uncertainty in differentiating between some of these shapes

* Overall, the implementations in this course helped me connect the "theory" of ML to the practical implementations, the problems that arise, the common fixes, the different approaches and the inside working of a library like sklearn
