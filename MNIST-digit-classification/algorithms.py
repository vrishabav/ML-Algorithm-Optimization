import numpy as np
from collections import Counter

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    X = X/255.0
    return X, y


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        Xc = X-self.mean_
        cov = np.cov(Xc.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        self.components_ = vecs[:, :self.n_components]

    def transform(self, X):
        return (X-self.mean_) @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        X_rec = X_pca @ self.components_.T
        return X_rec+self.mean_


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.n_classes_ = len(np.unique(y))

    def predict(self, X_test):
        preds = []
        for x in X_test:
            d = np.sqrt(np.sum((self.X_train_-x) ** 2, axis=1))
            k_idx = np.argsort(d)[: self.k]
            labels = [self.y_train_[i] for i in k_idx]
            preds.append(Counter(labels).most_common(1)[0][0])
        return np.array(preds)

    def predict_proba(self, X_test):
        probas = []
        for x in X_test:
            d = np.sqrt(np.sum((self.X_train_-x) ** 2, axis=1))
            k_idx = np.argsort(d)[: self.k]
            labels = [self.y_train_[i] for i in k_idx]
            counts = Counter(labels)
            p = np.zeros(self.n_classes_)
            for c, cnt in counts.items():
                if c < self.n_classes_:
                    p[int(c)] = cnt/self.k
            probas.append(p)
        return np.array(probas)


class OneVsRestClassifier:
    def __init__(self, base_classifier_class, **params):
        self.base_classifier_class = base_classifier_class
        self.params = params
        self.classifiers_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        for c in self.classes_:
            yb = (y == c).astype(int)
            clf = self.base_classifier_class(**self.params)
            print(f"  Training classifier for class {c}...")
            clf.fit(X, yb)
            self.classifiers_.append(clf)

    def predict_proba(self, X):
        probas = [clf.predict_proba(X) for clf in self.classifiers_]
        return np.stack(probas, axis=1)

    def predict(self, X):
        all_probas = self.predict_proba(X)
        return self.classes_[np.argmax(all_probas, axis=1)]


class PCAReconstructionClassifier:
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.models_ = []
        self.classes_ = None

    def fit(self, X_raw, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for c in self.classes_:
            print(f"  Training PCA for class {c}...")
            Xc = X_raw[y == c]
            pca_c = PCA(n_components=self.n_components)
            pca_c.fit(Xc)
            self.models_.append(pca_c)

    def predict(self, X_raw):
        errors = np.zeros((len(X_raw), len(self.classes_)))
        for i, pca_c in enumerate(self.models_):
            Xt = pca_c.transform(X_raw)
            Xr = pca_c.inverse_transform(Xt)
            errors[:, i] = np.sum((X_raw-Xr) ** 2, axis=1)
        return self.classes_[np.argmin(errors, axis=1)]

    def predict_proba(self, X_raw):
        errors = np.zeros((len(X_raw), len(self.classes_)))
        for i, pca_c in enumerate(self.models_):
            Xt = pca_c.transform(X_raw)
            Xr = pca_c.inverse_transform(Xt)
            errors[:, i] = np.sum((X_raw-Xr) ** 2, axis=1)
        scores = 1.0/(errors+1e-15)
        return scores/np.sum(scores, axis=1, keepdims=True)


class XGBoostTree:
    def __init__(self, max_depth=3, min_child_weight=1, gamma=0, reg_lambda=1.0, colsample=1.0, sketch_eps=0.1):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.colsample = colsample
        self.sketch_eps = sketch_eps
        self.tree = None
        self.col_indices = None
        self.sorted_features = None
    
    def _calculate_leaf_weight(self, G, H):
        return -G/(H+self.reg_lambda)

    def _calculate_node_score(self, G, H):
        if H < self.min_child_weight:
            return 0.0
        return (G**2)/(H+self.reg_lambda)

    def fit(self, X, gradient, hessian):
        n_samples, n_features = X.shape
        
        if self.colsample < 1.0:
            n_cols = max(1, int(n_features*self.colsample))
            self.col_indices = np.random.choice(n_features, n_cols, replace=False)
            X_sub = X[:, self.col_indices]
        else:
            self.col_indices = None
            X_sub = X
        
        self.sorted_features = None
        sample_mask = np.ones(n_samples, dtype=bool)
        self.tree = self._build_tree(X_sub, gradient, hessian, sample_mask, depth=0)
                
    def _build_tree(self, X_sub, gradient, hessian, sample_mask, depth):
        G, H = np.sum(gradient[sample_mask]), np.sum(hessian[sample_mask])
        
        if depth >= self.max_depth or np.sum(sample_mask) < 2 or H < self.min_child_weight:
            leaf_weight = self._calculate_leaf_weight(G, H)
            return {'leaf': leaf_weight}
        
        node_score = self._calculate_node_score(G, H)
        best_gain, best_split = 0.0, None

        g_node = gradient[sample_mask]
        h_node = hessian[sample_mask]
        
        for j in range(X_sub.shape[1]):
            feat_col = X_sub[sample_mask, j]
            sparse_mask = (feat_col == 0)
            dense_mask = ~sparse_mask

            g_sparse, h_sparse = g_node[sparse_mask], h_node[sparse_mask]
            g_dense_unsorted, h_dense_unsorted = g_node[dense_mask], h_node[dense_mask]
            feat_dense_unsorted = feat_col[dense_mask]
            
            if len(g_dense_unsorted) == 0:
                continue
            
            G_sparse, H_sparse = np.sum(g_sparse), np.sum(h_sparse)
            
            sort_idx_dense = np.argsort(feat_dense_unsorted)
            feat_dense = feat_dense_unsorted[sort_idx_dense]
            g_dense, h_dense = g_dense_unsorted[sort_idx_dense], h_dense_unsorted[sort_idx_dense]
            
            g_cumsum, h_cumsum = np.cumsum(g_dense), np.cumsum(h_dense)
            G_total, H_total = g_cumsum[-1], h_cumsum[-1]
            if H_total <= 1e-6:
                continue
            
            cum_weights = np.cumsum(h_dense)
            total_weight = cum_weights[-1]
            if total_weight == 0:
                continue

            n_candidates = max(int(1.0/self.sketch_eps), 3)
            quantiles = np.linspace(0, 1, n_candidates)
            target_weights = quantiles*total_weight
            
            candidate_positions = np.searchsorted(cum_weights, target_weights)
            candidate_positions = np.unique(np.clip(candidate_positions, 0, len(g_cumsum)-1))
            
            for pos in candidate_positions:
                if pos >= len(g_cumsum)-1: continue 
                if pos > 0 and feat_dense[pos] == feat_dense[pos-1]: continue

                G_left, H_left = g_cumsum[pos], h_cumsum[pos]
                G_right, H_right = G_total-G_left, H_total-H_left
                threshold = feat_dense[pos+1]
                
                gain_left = self._split_gain(G_left+G_sparse, H_left+H_sparse, G_right, H_right, node_score)
                gain_right = self._split_gain(G_left, H_left, G_right+G_sparse, H_right+H_sparse, node_score)
                
                if gain_left > best_gain or gain_right > best_gain:
                    sparse_left = gain_left > gain_right
                    best_gain = max(gain_left, gain_right)
                    best_split = {"feature_idx": j, "threshold": threshold, "sparse_left": sparse_left}

        if best_split is None or best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(G, H)
            return {"leaf": leaf_weight}
        
        j, threshold, sparse_left = best_split["feature_idx"], best_split["threshold"], best_split["sparse_left"]
        feat = X_sub[:, j]
        sparse_mask = (feat == 0)
        
        if sparse_left:
            left_mask = (feat < threshold) & sample_mask
        else:
            left_mask = (feat < threshold) & ~sparse_mask & sample_mask
        
        right_mask = sample_mask & ~left_mask
        
        left_tree = self._build_tree(X_sub, gradient, hessian, left_mask, depth+1)
        right_tree = self._build_tree(X_sub, gradient, hessian, right_mask, depth+1)
        
        return {"feature_idx": j, "threshold": threshold, "sparse_left": sparse_left, "left": left_tree, "right": right_tree}

    def _split_gain(self, G_L, H_L, G_R, H_R, parent_score):
        if H_L < self.min_child_weight or H_R < self.min_child_weight:
            return -np.inf
        score_L = self._calculate_node_score(G_L, H_L)
        score_R = self._calculate_node_score(G_R, H_R)
        gain = 0.5*(score_L+score_R-parent_score)
        return gain-self.gamma

    def predict(self, X):
        if self.col_indices is not None:
            X_sub = X[:, self.col_indices]
        else:
            X_sub = X
        return np.array([self._predict_one(x_sub, self.tree) for x_sub in X_sub])
    
    def _predict_one(self, x_sub, node):
        if "leaf" in node:
            return node["leaf"]
        val = x_sub[node["feature_idx"]]
        if val == 0:
            branch = "left" if node["sparse_left"] else "right"
        else:
            branch = "left" if val < node["threshold"] else "right"
        return self._predict_one(x_sub, node[branch])


class XGBoostClassifier:
    def __init__(self, n_estimators=40, learning_rate=0.3, max_depth=4, min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=0.5, reg_lambda=1.0, sketch_eps=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_score = 0.0
        self.sketch_eps = sketch_eps
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1/(1+np.exp(-x))
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        eps = 1e-15

        pos_ratio = np.clip(np.mean(y), eps, 1-eps)
        self.base_score = np.log(pos_ratio/(1-pos_ratio))
        pred = np.full(n_samples, self.base_score, dtype=float)
        self.trees = []
        
        for iteration in range(self.n_estimators):
            prob = self._sigmoid(pred)
            gradient = prob-y
            hessian = np.maximum(prob*(1-prob), eps)
            
            if self.subsample < 1.0:
                n_subsample = int(n_samples*self.subsample)
                sample_indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_sample, grad_sample, hess_sample = X[sample_indices], gradient[sample_indices], hessian[sample_indices]
            else:
                X_sample, grad_sample, hess_sample = X, gradient, hessian
            
            tree = XGBoostTree(
                max_depth=self.max_depth, 
                min_child_weight=self.min_child_weight, 
                gamma=self.gamma, 
                reg_lambda=self.reg_lambda, 
                colsample=self.colsample_bytree,
                sketch_eps = self.sketch_eps
            )
            tree.fit(X_sample, grad_sample, hess_sample)
            
            pred += self.learning_rate*tree.predict(X)
            self.trees.append(tree)
            
            if (iteration+1)%10 == 0:
                print(f"    Trained XGB Tree {iteration+1}/{self.n_estimators}")
    
    def predict_proba(self, X):
        pred = np.full(X.shape[0], self.base_score, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate*tree.predict(X)
        return self._sigmoid(pred)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


class VotingClassifier:
    def __init__(self, estimators):
        self.estimators_ = estimators
        self.n_classes_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))

    def predict_proba(self, X_dict):
        all_probas = []
        for name, clf in self.estimators_:
            if name == "KNN":
                all_probas.append(clf.predict_proba(X_dict['pca']))
            elif name == "PCA_Anomaly":
                all_probas.append(clf.predict_proba(X_dict['raw']))
            elif name == "XGB":
                all_probas.append(clf.predict_proba(X_dict['raw']))
        
        return np.mean(all_probas, axis=0)

    def predict(self, X_dict):
        probas = self.predict_proba(X_dict)
        return np.argmax(probas, axis=1)
