
import numpy as np
from typing import Dict, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def k_sparse_probe(latents: np.ndarray, labels: np.ndarray, ks: List[int]):
    """
    Compute k-sparse probing F1 scores.

    Args:
        latents: (N, L) latent activations (non-negative)
        labels:  (N,) 0/1 binary labels
        ks: list of k values

    Returns:
        f1_by_k: dict[int, float]
        order: feature indices sorted by |corr| desc
    """
    l = latents; y = labels.astype(np.float32)
    mu_l = l.mean(axis=0, keepdims=True)
    std_l = l.std(axis=0, keepdims=True) + 1e-6
    lz = (l - mu_l) / std_l
    corr = (lz * (y - y.mean())[:,None]).mean(axis=0) / (y.std()+1e-6)
    order = np.argsort(-np.abs(corr))

    f1 = {}
    for k in ks:
        idx = order[:k]; X = l[:, idx]
        if len(np.unique(labels)) < 2:
            f1[k] = 0.0; continue
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(X, labels); yhat = clf.predict(X)
        f1[k] = f1_score(labels, yhat)
    return f1, order

def splitting_score(f1_by_k: Dict[int, float], split_delta: float=0.05) -> Tuple[float, bool]:
    f1_1 = f1_by_k.get(1, 0.0); f1_2 = f1_by_k.get(2, f1_1)
    improve = f1_2 - f1_1
    return improve, (improve >= split_delta)
