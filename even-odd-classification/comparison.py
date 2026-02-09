import pandas as pd
import numpy as np
import time
import random
from collections import Counter

# metrics
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denominator = tp + fp
    if denominator == 0:
        return zero_division
    return tp / denominator

def recall_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = tp + fn
    if denominator == 0:
        return zero_division
    return tp / denominator

def f1_score(y_true, y_pred, zero_division=0):
    precision = precision_score(y_true, y_pred, zero_division)
    recall = recall_score(y_true, y_pred, zero_division)
    denominator = precision + recall
    if denominator == 0:
        return zero_division
    return 2 * (precision * recall) / denominator

def read_data(trainfile, validationfile):  
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'even'

    Xtrain = np.array(dftrain[featurecols])
    ytrain = np.array(dftrain[targetcol])

    Xval = np.array(dfval[featurecols])
    yval = np.array(dfval[targetcol])

    return (Xtrain, ytrain, Xval, yval)

# Logistic Regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=200, mini_batch_size=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.theta = None
        self.loss_history = []

    def _add_bias(self, x):
        try:
            m, n = x.shape
        except ValueError: # Handle 1D array
             x = x[:, np.newaxis]
             m, n = x.shape
        x_b = np.c_[np.ones((m, 1)), x]
        return x_b, m, n

    def _compute_loss(self, y, y_hat):
        m = y.shape[0]
        # Adding epsilon for numerical stability
        loss = - (y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
        return float(np.mean(loss))

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return np.where(
            z_clipped >= 0,
            1/(1 + np.exp(-z_clipped)),
            np.exp(z_clipped)/(1 + np.exp(z_clipped))
        )

    def fit(self, x, y):
        
        x_b, m, n = self._add_bias(x)
        y = y.reshape(-1, 1) # Ensure y is a column vector

        # Initialize theta
        self.theta = np.random.randn(n + 1, 1)
        self.loss_history = []

        n_batches = m // self.mini_batch_size

        for epoch in range(self.n_epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            x_shuffled = x_b[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                # Get mini-batch
                start_idx = i * self.mini_batch_size
                end_idx = start_idx + self.mini_batch_size
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Predictions for the batch
                linear = np.dot(x_batch, self.theta)
                predictions = self._sigmoid(linear)

                # Calculate errors and gradients
                errors = (y_batch - predictions)
                gradients = (-1.0/self.mini_batch_size) * np.dot(x_batch.T, errors)

                # Update weights (theta)
                self.theta -= self.lr * gradients

            # Compute and store loss for the epoch (on full training set)
            linear_full = np.dot(x_b, self.theta)
            predictions_full = self._sigmoid(linear_full)
            loss = self._compute_loss(y, predictions_full)
            self.loss_history.append(loss)

            if (epoch + 1) % 20 == 0:
                 print(f"  LR Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.4f}")

    def predict_proba(self, x):
        x_b, m, n = self._add_bias(x)
        linear = np.dot(x_b, self.theta)
        return self._sigmoid(linear).flatten() # Return 1D array of probabilities

    def predict(self, x):
        probabilities = self.predict_proba(x)
        return (probabilities >= 0.5).astype(int)


# Random Forest
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # index of feature to split on
        self.threshold = threshold          # threshold value to split
        self.left = left                    # left subtree
        self.right = right                  # right subtree
        self.value = value                  # class label for leaf nodes

    def is_leaf_node(self):
        # returns true if this node hold a value
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is an extra line from earlier
        # the indices of features to be used are passed as argument
        self.feature_indices = feature_indices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fit(self, X, y):
        # if subset of features is not defined
        if not self.feature_indices:
            self.feature_indices = list(np.arange(len(X[0])))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_classes = len(set(y))

        # base case 1
        if num_samples == 0:
            return None

        # base case 2
        # stopping conditions
        if (depth >= self.max_depth or
            num_classes == 1 or
            num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # greedy search for best split
        # feat_idxs instead of num_features in this line
        best_feat, best_thresh = self._best_split(X, y)

        # base case 3 - best split not found
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # recursive case
        # split is found
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        # if one of the parts is empty
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            # If either side is empty, return a leaf node with the most common label
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None

        n_candidates = 20

        for feat_idx in self.feature_indices:
            feature_values = X[:, feat_idx]
            unique_vals = np.unique(feature_values)

            if len(unique_vals) > n_candidates:
                percentiles = np.linspace(0.0, 100.0, n_candidates)
                thresholds = np.percentile(feature_values, percentiles)
                thresholds = np.unique(thresholds) # Remove duplicates
            elif len(unique_vals) > 1:
                thresholds = unique_vals
            else:
                continue

            for thresh in thresholds: # This loop is now much smaller
                gain = self._gini_gain(y, X[:, feat_idx], thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _gini_gain(self, y, feature_column, threshold):
        # parent gini
        parent_gini = self._gini(y)

        # generate splits
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold

        # If a split results in zero samples in one node, gain is 0
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        # weighted avg. gini of children
        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left/n) * gini_left + (n_right/n) * gini_right

        # gini gain
        return parent_gini - child_gini

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts/len(y)
        return 1.0 - sum(p**2 for p in probabilities if p > 0)

    def _most_common_label(self, y):
        if set(y) is None:
            return None
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.is_leaf_node():
            return node.value
        # recursive calling of left and right branches
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

class RandomForest:
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_features = len(X[0])
        
        k = int(np.sqrt(n_features))  # hardcoding taking sqrt of features

        for tci in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # selection feature indices
            feature_indices = random.sample(range(n_features), k)
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices = feature_indices
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if tci % 5 == 4:
                print(f"  RF Tree {tci+1}/{self.n_trees} completed")

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return np.array([X[i] for i in indices]), np.array([y[i] for i in indices])

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = list(zip(*tree_preds))
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]


# XGBoost
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
    
    def fit(self, X, y):
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
            
            if (iteration) % 10 == 9:
                print(f" XGB Tree {iteration+1}/{self.n_estimators} Trees completed")
    
    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.base_score, dtype=float)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    

def evaluate(model, model_name, Xtrain, ytrain, Xval, yval):
    print("Training " + model_name)

    # Training
    train_start = time.time()
    model.fit(Xtrain, ytrain)
    train_time = time.time() - train_start

    # Prediction
    pred_start = time.time()
    ypred = model.predict(Xval)
    pred_time = time.time() - pred_start

    # Metrics
    accuracy = accuracy_score(yval, ypred)
    precision = precision_score(yval, ypred, zero_division=0)
    recall = recall_score(yval, ypred, zero_division=0)
    f1 = f1_score(yval, ypred, zero_division=0)
    cm = confusion_matrix(yval, ypred)

    print(f"Training Time: {train_time:.2f}s")
    print(f"Prediction Time: {pred_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print()

    return {'Model': model_name, 'Training Time (s)': round(train_time, 2), 'Prediction Time (s)': round(pred_time, 2), 'Accuracy': round(accuracy, 4), 'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1 Score': round(f1, 4), 'Confusion Matrix': cm}


Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_test.csv')
print("Training samples: " + str(len(ytrain)))
print("Validation samples: " + str(len(yval)))
print()

results = []

# Logistic Regression
lr_model = LogisticRegression(n_epochs=100)
results.append(evaluate(lr_model, "Logistic Regression", Xtrain, ytrain, Xval, yval))

# Random Forest
rf_model = RandomForest(n_trees=100, max_depth=8, min_samples_split=10)
results.append(evaluate(rf_model, "Random Forest", Xtrain, ytrain, Xval, yval))

# XGBoost
xgb_model = XGBoostClassifier(n_estimators=100, max_depth=6, colsample_bytree=0.5, subsample=0.8, learning_rate=0.1, use_exact_greedy=False)
results.append(evaluate(xgb_model, "XGBoost", Xtrain, ytrain, Xval, yval))

print("Model Comparison\n")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
