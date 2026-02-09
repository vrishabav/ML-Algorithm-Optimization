import pandas as pd
import numpy as np
import time

# metrics
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Data reading
def read_data(trainfile, testfile):  
    dftrain = pd.read_csv(trainfile)
    dftest = pd.read_csv(testfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'even'

    Xtrain = np.array(dftrain[featurecols])
    ytrain = np.array(dftrain[targetcol])

    Xtest = np.array(dftest[featurecols])
    ytest = np.array(dftest[targetcol])

    return (Xtrain, ytrain, Xtest, ytest)

class XGBoostTree:
    def __init__(self, max_depth=3, min_child_weight=1, gamma=0, reg_lambda=1.0, colsample=1.0, use_exact_greedy=False, sketch_eps=0.1):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight  # minimum hessian sum in a child
        self.gamma = gamma  # minimum loss reduction in a split (complexity regularisation)
        self.reg_lambda = reg_lambda # L2 regularization
        self.colsample = colsample  # fraction of columns sampled
        self.use_exact_greedy = use_exact_greedy # use exact vs approximate algorithm
        self.tree = None
        self.col_indices = None
        self.sorted_features = None
        self.sketch_eps = sketch_eps
    
    def _calculate_leaf_weight(self, G, H):
        # Based on objective: G*w + 0.5*(H + lambda)*w^2 + alpha*|w|
        return -G/(H + self.reg_lambda)

    def _calculate_node_score(self, G, H):
        if H < self.min_child_weight:
            return 0.0
        return (G**2)/(H + self.reg_lambda)

    def fit(self, X, gradient, hessian):
        n_samples, n_features = X.shape
        
        # Column subsampling from section 2.3
        if self.colsample < 1.0:
            n_cols = max(1, int(n_features * self.colsample))
            self.col_indices = np.random.choice(n_features, n_cols, replace=False)
            X = X[:, self.col_indices]
        else:
            self.col_indices = None
        
        # pre-sort each feature for efficiently searching splits
        if self.use_exact_greedy:
            self.sorted_features = []
            for j in range(X.shape[1]):
                sorted_idx = np.argsort(X[:, j])
                self.sorted_features.append({'indices': sorted_idx, 'values': X[sorted_idx, j]})
        else:
            self.sorted_features = None # Not needed for approximate

        # recursive tree building
        sample_mask = np.ones(n_samples, dtype=bool)
        self.tree = self._build_tree(X, gradient, hessian, sample_mask, depth=0)
                
    def _build_tree(self, X, gradient, hessian, sample_mask, depth):
        # Calculate G and H for current node
        G, H = np.sum(gradient[sample_mask]), np.sum(hessian[sample_mask])
        
        # Stopping conditions
        if depth >= self.max_depth or np.sum(sample_mask) < 2 or H < self.min_child_weight:
            leaf_weight = self._calculate_leaf_weight(G, H)
            return {'leaf': leaf_weight}
        
        # Calculate node score using L1/L2-aware function
        node_score = self._calculate_node_score(G, H)
        best_gain, best_split = 0.0, None

        g_node = gradient[sample_mask]
        h_node = hessian[sample_mask]
        
        for j in range(X.shape[1]):
            if self.use_exact_greedy:
                sorted_idx = self.sorted_features[j]['indices']
                sorted_vals = self.sorted_features[j]['values']
                
                # Filter sorted lists by current node's sample_mask
                valid_mask = sample_mask[sorted_idx]
                valid_indices = sorted_idx[valid_mask]
                valid_values = sorted_vals[valid_mask]
                
                if len(valid_indices) == 0:
                    continue
                
                # Sparsity-Aware Split Finding (3.3)
                sparse_mask = (valid_values == 0) 
                dense_mask = ~sparse_mask
                
                # Get g/h values from the original gradient/hessian arrays
                g_sparse = gradient[valid_indices[sparse_mask]]
                h_sparse = hessian[valid_indices[sparse_mask]]
                
                g_dense = gradient[valid_indices[dense_mask]]
                h_dense = hessian[valid_indices[dense_mask]]
                feat_dense = valid_values[dense_mask]
                
                if len(g_dense) == 0:
                    continue 
                
                G_sparse, H_sparse = np.sum(g_sparse), np.sum(h_sparse)
                
                g_cumsum = np.cumsum(g_dense)
                h_cumsum = np.cumsum(h_dense)
                
                G_total, H_total = g_cumsum[-1], h_cumsum[-1]
                if H_total <= 1e-6:
                    continue
                
                # We only need to check splits bw distinct values
                for i in range(len(feat_dense) - 1):
                    if feat_dense[i] == feat_dense[i+1]:
                        continue
                    
                    pos = i # We split *after* index i
                    G_left, H_left = g_cumsum[pos], h_cumsum[pos]
                    G_right, H_right = G_total-G_left, H_total-H_left
                    
                    threshold = feat_dense[pos+1]
                    
                    # Evaluate gain sending sparse values left vs right
                    gain_left = self._split_gain(G_left + G_sparse, H_left + H_sparse, G_right, H_right, node_score)
                    gain_right = self._split_gain(G_left, H_left, G_right + G_sparse, H_right + H_sparse, node_score)
                    
                    if gain_left > best_gain or gain_right > best_gain:
                        sparse_left = gain_left > gain_right
                        best_gain = max(gain_left, gain_right)
                        best_split = {"feature_idx": j,
                            "threshold": threshold,
                            "sparse_left": sparse_left}

            else:
                feat_col = X[sample_mask, j]

                sparse_mask = (feat_col == 0)
                dense_mask = ~sparse_mask

                g_sparse = g_node[sparse_mask]
                h_sparse = h_node[sparse_mask]
                
                g_dense_unsorted = g_node[dense_mask]
                h_dense_unsorted = h_node[dense_mask]
                feat_dense_unsorted = feat_col[dense_mask]
                
                if len(g_dense_unsorted) == 0:
                    continue
                
                G_sparse, H_sparse = np.sum(g_sparse), np.sum(h_sparse)
                
                # Sort only the dense features for this node (local approximate)
                sort_idx_dense = np.argsort(feat_dense_unsorted)
                feat_dense = feat_dense_unsorted[sort_idx_dense]
                g_dense = g_dense_unsorted[sort_idx_dense]
                h_dense = h_dense_unsorted[sort_idx_dense]
                
                g_cumsum = np.cumsum(g_dense)
                h_cumsum = np.cumsum(h_dense)
                
                G_total, H_total = g_cumsum[-1], h_cumsum[-1]
                if H_total <= 1e-6:
                    continue
                
                cum_weights = np.cumsum(h_dense)
                total_weight = cum_weights[-1]
                
                if total_weight == 0:
                    continue

                n_candidates = max(int(1.0/self.sketch_eps), 3)
                quantiles = np.linspace(0, 1, n_candidates)
                target_weights = quantiles * total_weight
                
                candidate_positions = np.searchsorted(cum_weights, target_weights)
                candidate_positions = np.unique(np.clip(candidate_positions, 0, len(g_cumsum) - 1))
                
                for pos in candidate_positions:
                    # Skip last element, can't be a split point
                    if pos >= len(g_cumsum) - 1: 
                         continue 
                    if pos > 0 and feat_dense[pos] == feat_dense[pos-1]:
                         continue

                    G_left, H_left = g_cumsum[pos], h_cumsum[pos]
                    G_right, H_right = G_total - G_left, H_total - H_left
                    
                    threshold = feat_dense[pos+1]
                    
                    # Evaluate gain for both sparse directions (Section 3.3)
                    gain_left = self._split_gain(G_left + G_sparse, H_left + H_sparse, G_right, H_right, node_score)
                    gain_right = self._split_gain(G_left, H_left, G_right + G_sparse, H_right + H_sparse, node_score)
                    
                    if gain_left > best_gain or gain_right > best_gain:
                        sparse_left = gain_left > gain_right
                        best_gain = max(gain_left, gain_right)
                        best_split = {"feature_idx": j, "threshold": threshold, "sparse_left": sparse_left}

        # if no vaild split, leaf is returned
        if best_split is None or best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(G, H)
            return {"leaf": leaf_weight}
        
        j = best_split["feature_idx"]
        threshold = best_split["threshold"]
        sparse_left = best_split["sparse_left"]
        
        feat = X[:, j]
        sparse_mask = (feat == 0)
        
        # Samples assigned left/right
        if sparse_left:
            # Sparse values go left
            left_mask = (feat < threshold) & sample_mask
        else:
            # Sparse values go right
            left_mask = (feat < threshold) & ~sparse_mask & sample_mask
        
        right_mask = sample_mask & ~left_mask
        
        left_tree = self._build_tree(X, gradient, hessian, left_mask, depth + 1)
        right_tree = self._build_tree(X, gradient, hessian, right_mask, depth + 1)
        
        return {"feature_idx": j, "threshold": threshold, "sparse_left": sparse_left, "left": left_tree, "right": right_tree}

    def _split_gain(self, G_L, H_L, G_R, H_R, parent_score):
        if H_L < self.min_child_weight or H_R < self.min_child_weight:
            return -np.inf

        score_L = self._calculate_node_score(G_L, H_L)
        score_R = self._calculate_node_score(G_R, H_R)
        
        # Gain formula (equation 7)
        gain = 0.5 * (score_L + score_R - parent_score)
        
        return gain - self.gamma

    def predict(self, X):
        if self.col_indices is not None:
            X = X[:, self.col_indices]
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if "leaf" in node:
            return node["leaf"]

        val = x[node["feature_idx"]]
        if val == 0:
            # handle sparse value based on learned direction
            branch = "left" if node["sparse_left"] else "right"
        else:
            branch = "left" if val < node["threshold"] else "right"

        return self._predict_one(x, node[branch])


class XGBoostClassifier:
    def __init__(self, n_estimators=40, learning_rate=0.3, max_depth=4, min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=0.5, reg_lambda=1.0, use_exact_greedy=False, sketch_eps=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.use_exact_greedy = use_exact_greedy
        self.trees = []
        self.base_score = 0.0
        self.sketch_eps = sketch_eps
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1/(1 + np.exp(-x))
    
    def fit(self, X, y, X_test, y_test):
        n_samples = X.shape[0]
        eps = 1e-15

        # Base score defined as log odds
        pos_ratio = np.clip(np.mean(y), eps, 1 - eps)
        self.base_score = np.log(pos_ratio/(1 - pos_ratio))
        
        pred = np.full(n_samples, self.base_score, dtype=float)
        
        self.trees = []
        
        for iteration in range(self.n_estimators):
            prob = self._sigmoid(pred)
            
            gradient = prob - y
            hessian = np.maximum(prob*(1-prob), eps)
            
            # Subsampling
            if self.subsample < 1.0:
                n_subsample = int(n_samples * self.subsample)
                sample_indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_sample = X[sample_indices]
                grad_sample = gradient[sample_indices]
                hess_sample = hessian[sample_indices]
            else:
                X_sample = X
                grad_sample = gradient
                hess_sample = hessian
            
            # Weak learner
            tree = XGBoostTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight, gamma=self.gamma, reg_lambda=self.reg_lambda, colsample=self.colsample_bytree,use_exact_greedy=self.use_exact_greedy,sketch_eps = 0.1)
            tree.fit(X_sample, grad_sample, hess_sample)
            
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
            if (iteration) % 5 == 4:
                if X_test is not None and y_test is not None:
                    test_preds = self.predict(X_test) 
                    test_accuracy = accuracy_score(y_test,test_preds)
                    print(f" Test Accuracy after {iteration+1} Trees: {test_accuracy:.4f}")
    
    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.base_score, dtype=float)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


def evaluate(model, Xtrain, ytrain, Xtest, ytest):
    print("Training")

    # Training
    train_start = time.time()
    model.fit(Xtrain, ytrain, Xtest, ytest)
    train_time = time.time() - train_start

    # Prediction
    pred_start = time.time()
    ypred = model.predict(Xtest)
    pred_time = time.time() - pred_start

    # Metrics
    accuracy = accuracy_score(ytest, ypred)
    cm = confusion_matrix(ytest, ypred)

    print(f"Training Time: {train_time:.2f}s")
    print(f"Prediction Time: {pred_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    

Xtrain, ytrain, Xtest, ytest = read_data('MNIST_train.csv', 'MNIST_test.csv')
print(f"Training samples: {len(ytrain)}")
print(f"Test samples: {len(ytest)}\n")

xgb_model = XGBoostClassifier(n_estimators=100, max_depth=6, colsample_bytree=0.5, subsample=0.8, learning_rate=0.1, use_exact_greedy=False)
evaluate(xgb_model, Xtrain, ytrain, Xtest, ytest)
