
import numpy as np
from typing import Tuple, Optional

def spectral_norm_sq(W: np.ndarray, iters: int = 50) -> float:
    """
    Power iteration to estimate ||W||_2^2 for step size in ISTA.
    """
    m, n = W.shape
    v = np.random.randn(n); v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = W.T @ (W @ v)
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
    # Rayleigh quotient approximates largest eigenvalue of W^T W
    return float(v @ (W.T @ (W @ v)))

def ista_lasso(X: np.ndarray, D: np.ndarray, lam: float, steps: int = 200, step_size: Optional[float] = None, z0: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Solve min_z 0.5 ||x - D z||_2^2 + lam ||z||_1 for each row x in X, using ISTA with shared step size.
    X: (N, d), D: (d, m)
    returns Z: (N, m)
    """
    N, d = X.shape; m = D.shape[1]
    if z0 is None:
        Z = np.zeros((N, m), dtype=np.float32)
    else:
        Z = z0.copy()
        if Z.shape != (N, m):
            raise ValueError("z0 shape mismatch")
    if step_size is None:
        L = spectral_norm_sq(D)  # Lipschitz of grad
        step_size = 1.0 / (L + 1e-6)
    t = step_size
    Dt = D.T
    for _ in range(steps):
        R = X - Z @ Dt  # residual (N, d)
        G = Z + t * (R @ D)  # gradient step
        # soft-thresholding
        thr = lam * t
        Z = np.sign(G) * np.maximum(np.abs(G) - thr, 0.0)
    return Z

def lasso_objective(X: np.ndarray, D: np.ndarray, Z: np.ndarray, lam: float) -> np.ndarray:
    R = X - Z @ D.T
    return 0.5 * np.sum(R * R, axis=1) + lam * np.sum(np.abs(Z), axis=1)

def calibrate_lambda_by_dense_target(X: np.ndarray, D: np.ndarray, target_dense: float, steps: int = 100, lam_grid=None) -> float:
    """
    Choose lambda so that average activation rate (nonzero proportion) is close to target_dense.
    """
    if lam_grid is None:
        lam_grid = np.logspace(-5, -1, 9)
    best_lam = lam_grid[0]; best_diff = 1e9
    for lam in lam_grid:
        Z = ista_lasso(X, D, lam, steps=steps)
        act_rate = (Z != 0).mean(axis=0).mean()  # average across latents
        diff = abs(act_rate - target_dense)
        if diff < best_diff:
            best_diff = diff; best_lam = lam
    return float(best_lam)

def amortization_gap(X: np.ndarray, D: np.ndarray, Z_amort: np.ndarray, lam: float, ista_steps: int = 200) -> Tuple[float, float]:
    """
    Compute amortization gap: E[ L(z_amort) - L(z_star) ], where L is Lasso objective.
    Returns (mean_gap, median_gap).
    """
    Z_opt = ista_lasso(X, D, lam, steps=ista_steps)
    L_amort = lasso_objective(X, D, Z_amort, lam)
    L_opt = lasso_objective(X, D, Z_opt, lam)
    gap = L_amort - L_opt
    return float(np.mean(gap)), float(np.median(gap))
