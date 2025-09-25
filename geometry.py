
import numpy as np

def dictionary_geometry(W_dec: np.ndarray):
    W = W_dec / (np.linalg.norm(W_dec, axis=0, keepdims=True)+1e-9)
    C = W.T @ W; np.fill_diagonal(C, 0.0)
    return float(np.max(C)), float(np.min(C))
