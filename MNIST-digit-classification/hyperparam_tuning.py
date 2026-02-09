import numpy as np
import time
from algorithms import (load_data, PCA, KNNClassifier, OneVsRestClassifier, PCAReconstructionClassifier, XGBoostClassifier)
from sklearn.metrics import f1_score

X_train, y_train = load_data("MNIST_train.csv")
X_val, y_val = load_data("MNIST_validation.csv")

# Use a 20% subset for faster tuning
subset_size = int(0.2*X_train.shape[0])
X_sub = X_train[:subset_size]
y_sub = y_train[:subset_size]

# Use the full validation set for reliable scoring
print(f"Tuning on {subset_size} training samples, validating on {X_val.shape[0]} samples.")

best_params = {}

# PCA and KNN
print("\nPCA + KNN")
best_f1_knn = -1
best_pca_n = -1
best_knn_k = -1

for pca_n in [30, 40, 50]:
    print(f"Testing PCA n_components={pca_n}")
    pca = PCA(n_components=pca_n)
    X_sub_pca = pca.fit_transform(X_sub)
    X_val_pca = pca.transform(X_val)

    for k in [3, 5, 7]:
        print(f"  Testing KNN k={k}")
        knn = KNNClassifier(k=k)
        knn.fit(X_sub_pca, y_sub)
        
        y_pred = knn.predict(X_val_pca)
        f1 = f1_score(y_val, y_pred, average="weighted")
        print(f"    -> F1: {f1:.4f}")

        if f1 > best_f1_knn:
            best_f1_knn = f1
            best_pca_n = pca_n
            best_knn_k = k

best_params["pca_n_components"] = best_pca_n
best_params["knn_k"] = best_knn_k
print(f"Best KNN F1: {best_f1_knn:.4f} (pca_n={best_pca_n}, k={best_knn_k})")


# PCA Reconstruction Classifier
print("\nPCA Reconstruction Classifier")
best_f1_pr = -1
best_pr_n = -1

for n in [20, 30, 40, 50]:
    print(f"Testing PCA_Anomaly n_components={n}")
    pr = PCAReconstructionClassifier(n_components=n)
    pr.fit(X_sub, y_sub)
    
    y_pred = pr.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"  -> F1: {f1:.4f}")

    if f1 > best_f1_pr:
        best_f1_pr = f1
        best_pr_n = n
        
best_params["pca_recon_n"] = best_pr_n
print(f"Best PCA_Anomaly F1: {best_f1_pr:.4f} (n_components={best_pr_n})")


# XGBoost
print("\nXGBoost")
best_f1_xgb = -1
best_xgb_params = {}

# Grid search
for n_est in [30, 40]:
    for depth in [4, 5]:
        for lr in [0.2, 0.3]:
            
            params = {
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': depth,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 1.0,
                'colsample_bytree': 0.5,
                'reg_lambda': 1.0,
                'sketch_eps': 0.1
            }
            
            print(f"Testing XGB: n_estimators={n_est}, max_depth={depth}, lr={lr}")
            ovr_xgb = OneVsRestClassifier(XGBoostClassifier, **params)
            ovr_xgb.fit(X_sub, y_sub)
            
            y_pred = ovr_xgb.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="weighted")
            print(f"  -> F1: {f1:.4f}")
            
            if f1 > best_f1_xgb:
                best_f1_xgb = f1
                best_xgb_params = params

best_params["xgb"] = best_xgb_params
print(f"Best XGB F1: {best_f1_xgb:.4f}")
print(f"Best XGB Params: {best_xgb_params}")


# Summary
print("\nBest Parameters Found")
print(f"pca_n_components: {best_params['pca_n_components']}")
print(f"knn_k: {best_params['knn_k']}")
print(f"pca_recon_n: {best_params['pca_recon_n']}")
print("xgb: {")
for k, v in best_params['xgb'].items():
    print(f"    '{k}': {v},")
print("}")
