# **ML-Algorithm-Optimization**

**Author:** Vrishab Anurag Venkataraghavan

## **Overview**

This repository contains implementations and analyses of various Machine Learning optimization techniques and architectures. The projects range from comparing first-order vs. second-order optimization methods for Linear Regression to building custom Gradient Boosting frameworks and high-performance Voting Ensembles for digit classification.

## **Repository Structure**

ML-Algorithm-Optimization/  
├── Linear-Regression-Variations/    \# Optimization methods for Regression (SGD, Adam, SSHF, etc.)  
├── MNIST-even-odd-classification/   \# Custom XGBoost implementation for Binary Classification  
├── MNIST-digit-classification/      \# Voting Ensemble for Multi-class Classification  
├── MNIST\_train.csv                  \# Training dataset for classification tasks  
├── MNIST\_validation.csv             \# Validation dataset for classification tasks  
└── README.md

## **Project 1: Linear Regression Variations**

**Location:** /Linear-Regression-Variations

### **Description**

This module explores different optimization algorithms applied to Linear Regression using the diamonds.csv dataset. The goal was to compare convergence rates, stability, and final Mean Squared Error (MSE) between first-order and higher-order methods.

### **Methods Implemented**

1. **Stochastic Gradient Descent (SGD):** Baseline first-order method.  
2. **Adam (Adaptive Moment Estimation):** First-order method with adaptive learning rates and momentum.  
3. **ADMM (Alternating Direction Method of Multipliers):** Decomposes the problem into sub-problems (Ridge Regression \+ Quadratic Minimization).  
4. **Stochastic Quasi-Newton (SQN):** Uses L-BFGS recursion to approximate the inverse Hessian without forming the full matrix.  
5. **Sub-sampled Hessian-Free (SSHF):** A second-order method using Conjugate Gradient (CG) to solve the Newton system using Hessian-vector products.

### **Key Results**

The **Sub-sampled Hessian-Free (SSHF)** method was the unanimous winner, demonstrating the superior accuracy of second-order information.

| Algorithm | Type | Test MSE | Performance Note |
| :---- | :---- | :---- | :---- |
| **SGD** | First-Order | ![][image1] | Highly unstable; diverged without extensive tuning. |
| **Adam** | First-Order (Adaptive) | ![][image2] | Stable; converged quickly. |
| **ADMM** | First-Order (Splitting) | ![][image3] | Slightly outperformed Adam. |
| **SQN** | Quasi-Second-Order | ![][image2] | Comparable to Adam despite curvature approximation. |
| **SSHF** | Second-Order | ![][image4] | **Best Performer.** Lowest MSE across all sets. |

## **Project 2: MNIST Even-Odd Classification**

**Location:** /MNIST-even-odd-classification

### **Description**

A binary classification task to determine if a handwritten digit is Even or Odd. The core of this project is a **custom implementation of the XGBoost Classifier**, tuned via grid search and compared against Logistic Regression and Random Forest.

### **Custom XGBoost Implementation**

The implementation features:

* **Exact Greedy & Approximate Algorithms** for split finding.  
* **Sparsity Handling** optimized for the sparse nature of MNIST data (handling zero values efficiently).  
* **Hyperparameter Tuning** via grid search over 32 combinations.

### **Performance Comparison**

The XGBoost model significantly outperformed traditional classifiers in accuracy, though at a higher computational cost.

| Model | Accuracy | Training Time (s) | Precision | Recall | F1 Score |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression** | 0.8792 | **10.27** | 0.8801 | 0.8730 | 0.8765 |
| **Random Forest** | 0.9088 | 108.65 | 0.9433 | 0.8644 | 0.9032 |
| **Custom XGBoost** | **0.9632** | 158.17 | **0.9595** | **0.9658** | **0.9627** |

**Optimal Hyperparameters:**

* max\_depth: 6  
* subsample: 0.8  
* min\_child\_weight: 5  
* gamma: 0  
* reg\_lambda: 1.0

## **Project 3: MNIST Digit Classification**

**Location:** /MNIST-digit-classification

### **Description**

The final system architecture is a **Voting Ensemble** designed to classify all 10 MNIST digits. The system was optimized to balance high accuracy with a strict training/inference time constraint (under 5 minutes total runtime).

### **Architecture**

The ensemble averages probability distributions from three distinct models to maximize robustness:

1. **K-Nearest Neighbors (KNN):**  
   * Input: PCA-reduced data (30 components).  
   * ![][image5].  
   * *Role:* Fast, distance-based classification.  
2. **PCA Reconstruction Classifier (Anomaly Detection):**  
   * Method: Trains a separate PCA model (50 components) for each digit class.  
   * Prediction: Assigns the class with the lowest reconstruction error.  
   * *Role:* Unsupervised feature correlation.  
3. **XGBoost (One-vs-Rest):**  
   * Method: 10 binary classifiers trained on raw data.  
   * Config: n\_estimators=40, max\_depth=5, learning\_rate=0.2.  
   * *Role:* Strong gradient-boosted decision boundaries.

### **Final Results**

The ensemble strategy proved robust, correcting the individual errors of component models.

* **Total Runtime:** 254.64s (within the 300s limit).  
* **Ensemble Weighted F1 Score:** **0.9691**

| Model Component | Individual F1 Score | Train Time (s) | Predict Time (s) |
| :---- | :---- | :---- | :---- |
| **XGBoost (OvR)** | 0.9676 | 243.19 | 2.08 |
| **KNN (PCA)** | 0.9559 | \~0.00 | 2.97 |
| **PCA Anomaly** | 0.9383 | 1.05 | 0.16 |

## **Usage**

### **Dependencies**

Ensure you have the required Python packages installed:

pip install numpy pandas scikit-learn matplotlib xgboost

### **Running the Projects**

To run the final digit classification project:

cd MNIST-digit-classification  
python main.py

*Note: Ensure MNIST\_train.csv and MNIST\_validation.csv are located in the root directory for the scripts to access them correctly.*

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAUCAYAAAAa2LrXAAAEMUlEQVR4XtVXTUtVQRg+By2KooKyi/dj5p57L4EKEVyqRUIgFLgIIsFatGjXIlcG9QNa1SIqgqJdgYtIWiVKuJBq1c5QjMCFUUSESJARBtnznplzzsycb72X8oEXZ573Y2aeOzNntKwk2CbRStih8mY/DXnjVWwmNwb5Soajw0wicoXnCs6HXKX14FypueFVzztK3vj8CVZUjsaE3e2FrQwomuEZhJmNIFwlzCQhKTrJJxEbAgfnfIQz/hm2jvZatVo9LjwxII/i5YwNI+89zM0vFAq7Aq9lNRqNPah5E74VimGcDYPuUGPajvjVbA7NZnMbFvUc1lMqlcqumEKIq5Y7bPTIHlur1fYyxj7CTqHbUalUihDrjhJiof8B9orqE4/aPyD6NTUmDuEAkzH7EcgQskHYnVjMOBb3squra7fHgpuXIt5To03Afw72x1KmSMJDzAdodlIftS+AW6w6TsGLgv+MrH9fzd1ywOJ2YBETsCVYN3G0GrRn5AKnzePoxZTL5Z0QYopzRgKa8K9G+hGonviBBKsIOF8sFg8EaTooh46+yatAjUGdifg9IqiWwXGcw5jkWUvfRTNMLHDckjtJhRS5B7YMW6VjjDqnyaitxqL240BAAdy1TcqjfIzTo8abQMwQ6nKTJ5APNmvyhHZqlgpM6gsXAo6YPg/eLmLu7mWv0R6FPYOtWdqPwfwd6JHKDlxFu+nFRsOmO3PWFNEVj/HZqs//Q8n882a5i+vD5BZ52s5g7LoUYV3lSRBwC+6dB+DDsZ+LnSrq2W7MlMhllG8cQQNibjZ28nlPSPqLK6ShBxpI0zPNb2UK0UHHD4uaThOPwAIBV1Vefsk1Ybh4vozRcwbdDojxFP2flCt2YKapkoiXKK9crujiZUrPgY3UE+LxSXpumL4oKMdwSeXR75b8qDcP1D6EH+YFuDXYV9gA7DfsU6lUpKdNFigCpuy+EGwhiiKMrlH4f3MTiX75FnyEYznpfQTwnqvjjrnS29u73YwnQMB+KUKsgNSPGpjElx+pCQcvAdMfARIv3xHOhIjZRVBpoEt6HIuqycV3y2P4hI4pFZS7c6xadd5hIUeUvPtcvAN9QPij4BYceQfKmiSWLyhqvEV/rl6vH1RzY+Ye/ojYwRfY42NyFYQjVCbsNRAXICdCCwwZ7RQZM6hwD71c/EvWh136jXawrE+LvQEbEj3LonceF3ddP1EUi923goUfi56VzlGt4Eurg3w85hkTIGqMJOSIp6cFDx7Nvsnj5T8xuNhFb2DfwZ0IKtieuHOcrgDO6Tnzy3UoALcs34MUM4f2ST8iYb6I2we7mxRD42tEQuxmkaN0dGg0Kz8ojN/GHXqRni2mH0e1AiEuY7G3YAOmPxPiBv8/4B1AnY0gDJj+uDqtQxtL+9DGUB/JWw3+1N1GaxaSpUqWGMJf5ZL/d/ZiKoMAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAUCAYAAADFlsDIAAAENklEQVR4XtVXS2sUQRCeJREiikExrMnuTs8kMRAPKlnNQREPCiKiBz35AsVDfoCgFw+GXERED4JC1EMED4oXQfGQoFHEB4KngKAICqKomIAgmKBZv5qunumZ6XnsJop+UPRs1dfVVTXd1bOWlYVCVPHvo/GQG5+ZDfatlvDG+VjP5CNNF7KZiH8VCQFEimVCmm0u+N/8ziNMIUZ0JoqVqJ4jMrw6jnNKCDEFqdHY0dGxPMpJQqlUKmPOJZoLPw87Ozt7QgS5dgH2J+x/xhb2uRBHh8fPCDgX6veRPIMtSK4VCVxwXVeUy+WVeB52HPcV/Q5PYGgeK5XKOluIL8IWVzB3IYo1iPmTjuv2ByyvUEdh24vnJiouF7VV4zSOUIbmdHNQsoGgW5DIHSR8FT+blSfoZiHfbduuBmxehQfMHeCdMlYsFhcpkhD2NeimmeP5Jx7bvem8G5+mFSwtpzTbvEMtRklSspDxtra2xcpOhaIEUaydSkfQg4T9MnFQkBFNTQUaUcXB7izi+Y1eLAJ2Yjt032Bfo+sDyJXg6zB42002AuI7CF/7NKOG9JJ61gRKgtpbcCNkiz6bizUL/dakqaooXrE0Co6wXyyMVJR3sWJJfexlBJAOUUzQxCMZRwx0vO/i+C8jvjnKxpDbFxEpEch7iOsrQy4K4WJp0PQtdqxY/svwigXOgDY1BLUaOGvBfREUjH3YYg8dZ6blR+5KZKJAO+0AghuuVqsL0hxz33lJgh1QJJ28IOzPVAh1rHEJ7MLvX5YWpuM6h4iDW/G40mUBa/STYN4OxPjMTbqAspCSk0IOCgXkUDBTdLNFbWH4O2Q35Afe/mB3d/cSPJ9HsW7zTmomDhUdyV0EZxtNRJKrYb9BHMh+zWkmsJs+YM5r2m1RWwimbE26EBIIJjW9KdolkDNRWxpQiFWQB5g3gSQ2q2NINm8duVgTdG8hP2G/jrFP+D0xAg5ODuFIMSdfsRqEWs1UHx9d8ltrFEGc7KPjZ3mBHWkgKGq69OkQauhR8BH2e2JqcAy37mOYx2v9KKARn+UEqPG2o8+sx/hYNVDW34eMa7peyFcqjODrHfM24XkaMqScUxPH0ay53NcsWdAh4ipOHEGi8QYvAZ1s8Gk1Ce3SsK4h0KKccI2S0sbnCHApeacrXnHUdY++VhLyaH2E9FZwhWO8BxnTPza5wddUH8TzBsiklSNs2j3C8OmgaiD8T4c5IjMSgG4sLDiuChGRmxY3aQ56AjtwQtv+BSRxDPpP3KfoFhzD84pgBf/v1KiQf6NuYZyBnNA5CtGYHeNHKfMK/FEqDB+lUUcEk64eqEUjmjhM25lV6Hc9VDRb/jVqihIIdCMiqdN0JLu6uiq+weBQItFgpdtM6aTz80HzMW8+54Q/F0A+zxksZQ5oCRNi6pjCgDycOmFyadJ5iGdnQro1wG/G2/powPt6JwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAUCAYAAADFlsDIAAAEEUlEQVR4Xr1WS2tTQRROUEFpRXzE2DS5kxuULpUG3ahd+FiIKIgb0SrU/gRBt7oQBEVcKRRB0K0bQamoaFHxVdCNoNi6UERBkK4UrGj9vrkzN3Mnc19N6AeHzMz5zplzTs7MnULBRDEy6x7m6VebzdN8IaFC7ChSt3Fk1bWN2ywf8vqQ/GL34wgR6yxWkQP5fWSxMDlZ+AsA6x/qEOpPz4wkaqwuVuFAvV4/J4SYgczxt1KprJGKDE7qAW4p2+e+7w8V2y2L0L1XnFnwL1n6FmxLIm/FOoJrH7XWaDRWIIHLSFJUq9UNGI8hmQ+c21x7DM5G8Kc8zxvGdBHGByA/MD/WYslCnahVa/s5xh6rMH9Y6e+vGpxk6GK58lAIVN3paud/g6IsReB3INcxXazXMf/nCfETSTcNegTg9EE+gbMvXMQGtIF81ksYnxLCmws5QGlNqRe27/y6XzbXTdixzgtdcULAUblc7kHQDyATpVKpV6sEC4UjExTCvaMuFuR2eW25R7Ngs43rHBt/xi/DVK//ITfOPwHeCOLYEy5YVNgfhZ/D5lq8NwdykQtyw62QneYaiyW7y/N2yQWHU3V8X4jgHprmWrPZXAKba5ArnPu+X4buI/1FreUetDtir2twS9iDIp6GcYSaYADdOI+1M0ADydoWsvIiUIl8gfi2zgT0ByF/Ff+8uujHWUjqvVb3OYvFI5oWInxuAvd1tGDB3v2895LNk5HFVnKcxCI7bRiBjA2iS2ytjWZzkJ10kh8EVTDKrCcveOmrKTzZpQnFyoCi7LItFNjthd1Ldp1JcKaTE7l8qGBm0NrLbJ0NHjkWFYV6DFmH8XEWShXsNzlIqoHxt24Ui4DNV8gUuy1K6AZylApfJsQh3kEu2DoN0x0S3c3C1mq17XoNSdSxNslCkN5+DFsechVLIb5YORJNRYovdVnfRxCn2TFcw3y0PagWoL/KwiDhlZzrLVTBPvLrOjAwsBzjJ61iheDlrL62ydB+fb+ecAxbSEk1RFZeCGXADrjIy1IEndBXq1U34/eZvEAL4TPhEWSivxKssSuE4yPAIwzdXe0dtTvEwpgcJFomRx/3SOBWFn7cBe+pC95CriLkIhfCL9qcFr6v1HjSE0HXsANCvXp78Z9VF/sNvteUO3bMKOSt9s8OA++e/kIS8DGM48sXfSK4h2h7OoQwng4xyFWMFDITwYYTuhCW3IS9fNWroN9SODa6hv/6NOQ7ZAzyRjjuFCS0Hmuv8JI/Av1ZETxSU6KT/keE+Si14DkepbHQu5m7FjME0T0U5VcRBRxCYmf4q+88gyKBpHawUOwqHJ/VEc5CwCqW/MlcqczEGKTaJxASVHnQ7sZqm1yI4adXNVHZATp5PM7fMhWugljT/0R++G6o5uo4AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAUCAYAAADFlsDIAAAD8ElEQVR4Xs1XPWgUQRTeJREMCqIkOXM/O7uJWliJ5w8i2GgjEovYiGghorEXDAkW2llICguVcCJaasAqpFAwiAgSsIoIaiNIBJsDQdHCxO/tzmRnZ9/O7pkl5oPH7nzvd97NzM45joSrXli4eQbrCuukVFmGUU2x4taq4WuSpAj4ZiVg060zcKVyXAlIh00zPJdAroHEP63mDpAImLMofN+/KYRoQ5bpWa3Wek2bLAwODu6Czwz5Is5Lz/N2R5pkNuKl3R+Ser2+LWHgJD2yak0h1zDXoDgw2S0o/k4QBAIT2In3Kd8PPtDYtDXRaDT2wf4bZBLDLjRrXAjvFxozbJi6sPkB/mKz2dyA51ZPiBb47lBpGOtMWsdzCjbdqoDJbZS/9iNkCQsnYLwkJ9fU7XWQjmwQ4yGNVZHgx8hf2Q0NDfVjvEANUhyafBLcU8qvuPxp5ukLwli+6agZ67u/UtmEop9D5vr6+jYrJTUBssyskBVgoqNkA5ngeG18Wh+HiNJ0JTgnXSZ8z8P3uEZLTQTUd0544kyS7QCdOiHhYcjRaOSG/rJZS+CP6bZ6bOhbsqFjGk3xhqPmuN30w2H8gMa0/YIgOEJ6PCu6j4pr1g47uIpXZh0h3HBrz5pnXxjDDFQ6tARyxXyBBDGbBG0/vVnKXTWLViptPbzPy3hzkGuQ25Cf1DwtXCaQZw9Wz1uzYYhxqlqr1XUuhdKbZgREUWdRyNRemowlGSZAk18WGSuLmoXnAOQzjR15mBNqmKSgH8PTfwxLMidcZQdI4HcCOd7QqjNtCPYoJUIW08bS7jF1JmA3TU3I2oZMs1aAPBUvWm0jOh8jnrKx9RchH2m1aXQCXLM4blWgXwp4D7ll6iIkU/pBchsq6M2qVqu9eH9nNks2UvoWmIo0KdKsGOmG60yBrDzkXesZiriuzhKML9iKgn6CaxbuWSOqOX58NbE0K4Y+gcRkXGYb+vw25FC0MaydUYiLLTEpzxHaNgO4B+3H8zVxZCL5F5A5jaOlSB+BGWqKCodxC/Ha2vgQ5LeedOXMsnxAFMIDXvAHvKolE2FOowVsRyzQ7SlpeH544Rmiyzx9zcjGE9HWIqFtpiJgfAXyHRM6SAzOuh0YL8LmropPKxXcYz/wfcXRRwQ+l9U4C4Ht6uDwV4dIYxIlQG6H6KuWlmlH3upl0Qto6gK9K/96vdGDidyDro3nJTw/QWZpW8dZwtWxHfxXyH3IE1Hw6uCzl1IJV15KRXQp1ejMZmXQFkVBFPOPSqM/0yj8Kj0VaQL6JiZ/AzKKv0CN2Ei9sW4M0p48MrQczXFWdOzQCWzBeR3PriH+ewFlovTJ2AP+BdwF9OrPszsqAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAADEElEQVR4XsVWO4gUQRDdwQ38oqDr4n66Z3bFDUQMVAQxMPDA5BDE4EBzQYwORDAy0NDEeJNDDhE21EAQbkUQNDMQTIRTNBIzMZG781V/dnp6unt69lAfFD1d71VX9Wd6ppE0/jVcGV0+n9eFCKVXognVenWNMFeGpRZdw5f3c2e9BPXhGt/lcyJaWAVrxZ1wcS6fBb8EDGPsFewL5/xnmqZPbYkfrq20vP7MlUA9R8hQ087cawwI4jIEq7At2HJ1Ng/vcddFu93egzrGqGsF7WvYJufsSVGFZJ1O5xDID7D1Xq/XNbl5IMPmC1ar/JlhEalPo+A0LNKiZll2MleC6ff7Z0D8gk3gaeakCVchLl8NOMI73dkibmmBLhztqVwJLkvTG4q4o92tVmsvWUHoQNlb9pSgJabUeKZVx4qfM/q3qL5ut3tQOLQWoglnfBOFX1TCMewj7BNjfEEPEAs5rhrdNQ+Xz4MOisVZf4daNopMIgr9jcKf43wfxfMqzQzte5olgpYCiZrgL2DCV2MtyzJuD1JAnqtJetTwiMkb77ahkqACYWuwZ3pgmgSTZ2qHVPmr3x4S58kxgTquU42DwWC/6U/oDVbFf4PdNwczh/MNTITgvIIcEZICSI8T0MN1+BW7e29GqKtwfTgc9kFcwiS+Z1ka3s4oqBJjZ6SAnd6FFV5Ae4L6FEmXBGqcwl7SHS+c/Z64CifoiGsQz3cRuEjPmMhZbrzdBSRi0ruhfaF2K9au+OZBbvAPle6H9o1Go31cfoims5sOnTGSn9fB6D+gsw07Tp9/IfQkmkHzVbqGtQGmPpFdWm0ufz2WNKXPeCbev0RtAeNTHJPDWkRbBNEaxG/xYTqt/SVYSePgEpZ8CfLfZPKDuAx7jBo30L4pqDCzAwUHgJm1iz82EnmKUrICJGtpwt0S5P8Ku0Yrj9vkWINCqoLiNAS3yu31oZ66EuHhrLUPixtCECeMkvwl6MxmBaFqLK7QrREXB72C/wmVuSsFIYSDw6yFWuJ5IBK4jkq5G4vIsEjZDCF9iNse/gAeUJODlWpX5QAAAABJRU5ErkJggg==>
