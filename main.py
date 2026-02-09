import numpy as np
import time
from algorithms import (load_data, PCA, KNNClassifier, OneVsRestClassifier, PCAReconstructionClassifier, XGBoostClassifier, VotingClassifier)
from sklearn.metrics import f1_score, classification_report, confusion_matrix

X_train, y_train = load_data("MNIST_train.csv")
X_val, y_val = load_data("MNIST_validation.csv")

if X_train is None:
    print("Could not load data.")

start_total = time.time()

params = {
    "pca_n_components": 30,
    "knn_k": 3,
    "pca_recon_n": 50,
    "xgb": {
        'n_estimators': 40,
        'learning_rate': 0.2,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.5,
        'reg_lambda': 1.0,
        'sketch_eps': 0.1
    }
}

# PCA for KNN
pca = PCA(n_components=params["pca_n_components"])
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

X_val_dict = {"pca": X_val_pca, "raw": X_val}
estimators = []
models = {}
train_times = {}

# KNN on PCA
print("\nTraining KNN")
t0 = time.time()
knn = KNNClassifier(k=params["knn_k"])
knn.fit(X_train_pca, y_train)
train_times["KNN"] = time.time()-t0
estimators.append(("KNN", knn))
models["KNN"] = knn
print()

# PCA Anomaly
print("PCA Reconstruction")
t0 = time.time()
pr = PCAReconstructionClassifier(n_components=params["pca_recon_n"])
pr.fit(X_train, y_train)
train_times["PCA_Anomaly"] = time.time()-t0
estimators.append(("PCA_Anomaly", pr))
models["PCA_Anomaly"] = pr
print()

# XGBoost
print("XGBoost")
t0 = time.time()
ovr_xgb = OneVsRestClassifier(XGBoostClassifier, **params["xgb"])
ovr_xgb.fit(X_train, y_train)
train_times["XGB"] = time.time()-t0
estimators.append(("XGB", ovr_xgb))
models["XGB"] = ovr_xgb
print()

print("\nIndividual Model Performance on Validation Set")
for name, model in models.items():
    t0_pred = time.time()
    if name == "KNN":
        y_pred_ind = model.predict(X_val_pca)
    else: # PCA Anomaly and XGB
        y_pred_ind = model.predict(X_val)
    t_pred = time.time() - t0_pred
    
    f1_ind = f1_score(y_val, y_pred_ind, average="weighted")
    t_train = train_times[name]
    print(f"  {name:12s} F1={f1_ind:.4f}   Train={t_train:.2f}s   Test={t_pred:.2f}s")


# Ensemble
print("\nEnsemble")
ensemble = VotingClassifier(estimators=estimators)
ensemble.fit(None, y_train) # sets the class count

start_pred = time.time()
y_pred_ens = ensemble.predict(X_val_dict)
end_pred = time.time()
t_pred_ens = end_pred - start_pred

f1_ens = f1_score(y_val, y_pred_ens, average="weighted")
end_total = time.time()

print(f"\nEnsemble Model (Weighted F1): {f1_ens:.4f}   Test={t_pred_ens:.2f}s")

print("\nClassification by Ensemble:")
print(classification_report(y_val, y_pred_ens))

print("Confusion Matrix (Ensemble):")
print(confusion_matrix(y_val, y_pred_ens))

print(f"\nTotal runtime: {end_total - start_total:.2f}s")