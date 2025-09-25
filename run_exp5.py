# coding: utf-8
"""
Evaluate SAE checkpoints on collected activations (Gemma-2-2B L12 by default).

Unified metrics protocol:
- All metric statistics (Dead/Dense, k-sparse F1, ΔF1, Absorption) use nonnegative codes
  Z_relu = max(Z, 0) for consistency with "firing" semantics.
- k-sparse probe: z-score per column on Z_relu, rank features by |corr(Zs_j, y)| (global),
  train LR on the standardized features (class_weight='balanced'), report F1@k and ΔF1=F1@2-F1@1.
- Absorption: main = argmax |corr| (global), aux = next top-K by |corr|;
  on positive examples, count cases where main did NOT fire (<=0) while any aux DID fire (>0).
- NMSE uses Z (not ReLU'd), i.e., X_hat = Z @ D^T.
- Amortization Gap: ISTA-200 on L1 objective with optional lambda calibration.

Writes: outputs/metrics.parquet

Run:
  python -m src.exp5.run_exp5 --config config.yaml
"""

import os
import re
import math
import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import torch
except Exception:
    torch = None

try:
    from safetensors import safe_open
except Exception:
    safe_open = None

# sklearn optional (for F1/Absorption via simple probing)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------
# IO & config
# ---------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_activations(output_dir: str, layer_index: int) -> np.ndarray:
    path = os.path.join(output_dir, f"acts_layer{layer_index}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"activation file not found: {path}")
    acts = np.load(path, mmap_mode="r")
    # acts: [num_sequences, seq, hidden_dim]
    if acts.ndim != 3:
        raise ValueError(f"Expected activations shape [N, S, D], got {acts.shape}")
    N, S, D = acts.shape
    X = acts.reshape(N * S, D).astype(np.float32, copy=False)
    return X


# ---------------------------
# SAE weight loading & orientation
# ---------------------------

def _np(t):
    # map to numpy array regardless of torch/tensor type
    if torch is not None and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def _try_keys(d: Dict[str, Any], keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _load_pt_or_safetensors(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if safe_open is None:
            raise RuntimeError("safetensors not installed but .safetensors file provided.")
        weights = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
        return weights
    # fallback to torch.load for .pt / others
    if torch is None:
        raise RuntimeError("PyTorch not installed but .pt file provided.")
    obj = torch.load(path, map_location="cpu")
    # state dict wrapper
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj if isinstance(obj, dict) else {"weight": obj}

def _extract_sae_weights(raw: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Try many common key names across repos:
    - encoder/decoder (weight/bias)
    - W_enc, W_dec, b_enc, b_dec
    - ae.* nesting
    """
    # flatten nested dicts lightly
    cand = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                cand[f"{k}.{kk}"] = vv
        cand[k] = v

    # common pools
    W_enc = _try_keys(cand, [
        "W_enc", "encoder.weight", "enc.weight", "ae.W_enc",
        "W_enc.weight", "E", "proj", "proj.weight", "linear.weight",
        "model.encoder.weight"
    ])
    W_dec = _try_keys(cand, [
        "W_dec", "decoder.weight", "dec.weight", "ae.W_dec",
        "W_dec.weight", "D", "dict", "dictionary", "linear_dec.weight",
        "model.decoder.weight"
    ])
    b_enc = _try_keys(cand, [
        "b_enc", "encoder.bias", "enc.bias", "ae.b_enc",
        "bias", "linear.bias", "model.encoder.bias"
    ])

    if W_enc is None:
        # some repos use "W" for encoder and "D" for decoder
        W_enc = _try_keys(cand, ["W"])
    if W_dec is None:
        W_dec = _try_keys(cand, ["D"])

    if W_enc is None or W_dec is None:
        raise KeyError(f"Cannot find W_enc/W_dec keys in checkpoint (keys: {list(cand.keys())[:20]}...)")

    W_enc = _np(W_enc)
    W_dec = _np(W_dec)
    b = None if b_enc is None else _np(b_enc).reshape(1, -1)
    return W_enc, W_dec, b

def _canon_orientations(W_enc: np.ndarray, W_dec: np.ndarray, d_in: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize to:
      W_enc: [d_in, n_lat]
      D    : [d_in, n_lat]
    Accepts incoming shapes [n_lat, d_in] or [d_in, n_lat].
    """
    if W_enc.shape[0] == d_in:
        Wenc = W_enc
    elif W_enc.shape[1] == d_in:
        Wenc = W_enc.T
    else:
        raise ValueError(f"W_enc shape {W_enc.shape} incompatible with d_in={d_in}")

    if W_dec.shape[0] == d_in:
        D = W_dec
    elif W_dec.shape[1] == d_in:
        D = W_dec.T
    else:
        raise ValueError(f"W_dec shape {W_dec.shape} incompatible with d_in={d_in}")

    if Wenc.shape[1] != D.shape[1]:
        raise ValueError(f"latent dim mismatch: W_enc {Wenc.shape}, D {D.shape}")
    return Wenc.astype(np.float32, copy=False), D.astype(np.float32, copy=False)


# ---------------------------
# SAE forward (amortized)
# ---------------------------

def relu_inplace(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)

def topk_mask_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a sparse mask (bool) keeping top-k per row; fallbacks to full if k <= 0 or k>=m."""
    n, m = scores.shape
    if k <= 0 or k >= m:
        return np.ones_like(scores, dtype=bool)
    # argpartition for top-k indices per row
    idx = np.argpartition(scores, kth=m-k, axis=1)[:, -k:]
    mask = np.zeros_like(scores, dtype=bool)
    row_idx = np.arange(n)[:, None]
    mask[row_idx, idx] = True
    return mask

def encode_amortized(X: np.ndarray, W_enc: np.ndarray, b_enc: Optional[np.ndarray], variant: str,
                     target_dense: float = 0.1, guessed_k: Optional[int] = None) -> np.ndarray:
    """
    Standard: z = ReLU(X @ W_enc + b)
    Top-K   : scores = X @ W_enc + b ; keep top-k per row (may include negatives).
              For metrics we will ReLU later (Z_relu = max(Z,0)).
    """
    Z_lin = X @ W_enc
    if b_enc is not None:
        if b_enc.shape[1] != Z_lin.shape[1]:
            if b_enc.ndim == 2 and b_enc.shape[0] == 1 and b_enc.shape[1] != Z_lin.shape[1]:
                b_enc = b_enc[:, :Z_lin.shape[1]]
            else:
                b_enc = b_enc.reshape(1, Z_lin.shape[1])
        Z_lin = Z_lin + b_enc

    if "topk" in variant.lower():
        m = Z_lin.shape[1]
        if guessed_k is None:
            guessed_k = max(1, int(round(target_dense * m)))
        mask = topk_mask_rows(Z_lin, guessed_k)
        Z = np.where(mask, Z_lin, 0.0).astype(np.float32, copy=False)  # keep signed scores for recon; metrics will use Z_relu
        return Z
    else:
        Z = Z_lin.astype(np.float32, copy=False)
        return relu_inplace(Z)  # Standard: amortized codes are ReLU'ed


# ---------------------------
# Metrics (unified protocol)
# ---------------------------

def nmse(X: np.ndarray, X_hat: np.ndarray, eps: float = 1e-9) -> float:
    num = np.mean(np.sum((X - X_hat)**2, axis=1))
    den = np.mean(np.sum(X**2, axis=1)) + eps
    return float(num / den)

def dead_and_dense_rates(Z_relu: np.ndarray, dense_thresholds=(0.1, 0.2), dead_thr: float = 1e-6):
    """
    Z_relu: nonnegative codes used for "firing" semantics.
    """
    active = (Z_relu > 0.0).astype(np.float32)
    freq = active.mean(axis=0)  # fraction of samples where latent fires
    dead = float((freq <= dead_thr).mean())
    dense = {f"dense@{t:.1f}": float((freq >= t).mean()) for t in dense_thresholds}
    return dead, dense

def pseudo_labels_from_norm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    th = np.median(norms)
    y = (norms > th).astype(np.int32)
    return y

def _zscore_cols(Z_relu: np.ndarray) -> np.ndarray:
    mu = Z_relu.mean(axis=0, keepdims=True)
    std = Z_relu.std(axis=0, keepdims=True) + 1e-6
    return (Z_relu - mu) / std

def _corr_abs_order(Zs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson corr with y (binary 0/1); return indices sorted by |corr| desc.
    """
    y0 = y.astype(np.float32)
    y_center = y0 - y0.mean()
    num = (Zs * y_center[:, None]).mean(axis=0)
    den = (Zs.std(axis=0) * (y0.std() + 1e-6)) + 1e-6
    corr = num / den
    order = np.argsort(-np.abs(corr))
    return order

def k_sparse_probe_corr(Z_relu: np.ndarray, y: np.ndarray, ks=(1,2,3,5)) -> Tuple[Dict[str, float], np.ndarray]:
    """
    - Standardize Z_relu per column to Zs
    - Rank by |corr(Zs_j, y)|
    - Train LR on Zs[:, idx[:k]] with class_weight='balanced'
    - Return F1@k and ΔF1 (= F1@2 - F1@1), and the feature order.
    """
    out = {f"F1@{k}": float("nan") for k in ks}
    out["dF1"] = float("nan")

    if Z_relu.size == 0 or len(np.unique(y)) < 2:
        return out, np.arange(0)

    Zs = _zscore_cols(Z_relu)
    order = _corr_abs_order(Zs, y)

    if not SKLEARN_OK:
        return out, order

    for k in ks:
        idx = order[:k]
        Xk = Zs[:, idx]
        clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
        try:
            clf.fit(Xk, y)
            yhat = clf.predict(Xk)
            out[f"F1@{k}"] = float(f1_score(y, yhat))
        except Exception:
            out[f"F1@{k}"] = float("nan")

    if all(np.isfinite([out.get("F1@1", np.nan), out.get("F1@2", np.nan)])):
        out["dF1"] = float(out.get("F1@2", np.nan) - out.get("F1@1", np.nan))

    return out, order

def absorption_from_corr(Z_relu: np.ndarray, y: np.ndarray, order: np.ndarray, topk: int = 5) -> float:
    """
    Global correlation-based absorption:
      main = order[0]
      aux  = order[1:1+topk]
      On positives: main<=0 and any(aux>0)
    """
    if Z_relu.size == 0 or order.size == 0:
        return float("nan")
    pos = (y == 1)
    if not np.any(pos):
        return float("nan")
    main = int(order[0])
    aux = order[1:1+max(0, topk)]
    Zp = Z_relu[pos]
    main_on = Zp[:, main] > 0
    aux_on = (Zp[:, aux] > 0).any(axis=1) if aux.size > 0 else np.zeros(Zp.shape[0], dtype=bool)
    absorbed = (~main_on) & aux_on
    return float(absorbed.mean())


# ---------------------------
# ISTA & Amortization gap
# ---------------------------

def soft_threshold(x: np.ndarray, lmbd: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0.0)

def ista_lasso(X: np.ndarray, D: np.ndarray, lmbd: float, steps: int = 200, step_size: Optional[float] = None) -> np.ndarray:
    """
    Solve min_z 0.5||x - zD^T||^2 + lambda||z||_1 (row-wise), D: [d_in, n_lat]
    Returns Z of shape [n, n_lat]
    """
    n, d = X.shape
    d_in, n_lat = D.shape
    assert d == d_in
    # Lipschitz const of grad wrt Z is ||D D^T||
    if step_size is None:
        # crude but stable: 1 / spectral_norm(D)^2 (power iteration)
        Dt = D.T
        v = np.random.randn(n_lat).astype(np.float32)
        for _ in range(5):
            v = Dt @ (D @ v)
            v /= (np.linalg.norm(v) + 1e-8)
        L = float(v @ (Dt @ (D @ v)) + 1e-8)
        step_size = 1.0 / L
    Z = np.zeros((n, n_lat), dtype=np.float32)
    Dt = D.T
    for _ in range(steps):
        R = Z @ Dt  # [n, d_in]
        G = (R - X) @ D  # grad wrt Z: (zD^T - x)D
        Z = soft_threshold(Z - step_size * G, step_size * lmbd)
    return Z

def calibrate_lambda(X: np.ndarray, D: np.ndarray, target_dense: float = 0.1, steps: int = 200) -> float:
    """
    Simple grid search over lambda to match target avg density (mean fraction of nonzeros per sample).
    """
    grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 3e-2]
    best_l, best_diff = grid[0], 1e9
    for lmbd in grid:
        Z = ista_lasso(X, D, lmbd, steps=steps//4)  # quick eval
        dens = float((Z > 0).mean(axis=1).mean())  # average nonzero fraction per example
        diff = abs(dens - target_dense)
        if diff < best_diff:
            best_l, best_diff = lmbd, diff
    return best_l

def amortization_gap(X: np.ndarray, Z_amort: np.ndarray, D: np.ndarray, lmbd: float, steps: int = 200) -> Tuple[float, float]:
    """
    Gap = L(z_amort) - L(z_opt), where L(z)=0.5||x - zD^T||^2 + lambda||z||_1
    """
    Z_opt = ista_lasso(X, D, lmbd, steps=steps)

    def obj(X, Z, D, l):
        R = X - Z @ D.T
        recon = 0.5 * np.sum(R * R, axis=1)
        l1 = l * np.sum(np.abs(Z), axis=1)
        return recon + l1

    J_a = obj(X, Z_amort, D, lmbd)
    J_o = obj(X, Z_opt, D, lmbd)
    gap = J_a - J_o
    return float(np.mean(gap)), float(np.median(gap))


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = cfg.get("output_dir", "outputs")
    ensure_dir(output_dir)

    # model/layer
    layer_index = int(cfg.get("layer_index", cfg.get("model", {}).get("layer_index", 12)))

    # dataset activations
    X = load_activations(output_dir, layer_index)     # [Ntokens, d_in]
    Ntokens, d_in = X.shape  # <-- FIX: shape is a property, not callable

    # metrics config
    mcfg = cfg.get("metrics", {})
    dense_thresholds = tuple(mcfg.get("dense_thresholds", [0.1, 0.2]))
    dead_thr = float(mcfg.get("dead_threshold", 1e-6))
    corr_sub = int(mcfg.get("correlation_subsample", 50000))
    # allow ks to include more entries; we will intersect with available later
    probe_ks = tuple(mcfg.get("probe_ks", [1, 2, 3, 5]))
    absorption_topk = int(mcfg.get("absorption_topk", 5))
    normalize_mse = bool(mcfg.get("normalize_mse", True))

    # amortization gap config
    ag_cfg = mcfg.get("amortization_gap", {})
    ag_enabled = bool(ag_cfg.get("enabled", True))
    ag_steps = int(ag_cfg.get("ista_steps", 200))
    ag_lambda = ag_cfg.get("l1_lambda", None)
    ag_calib = bool(ag_cfg.get("calibrate_lambda", True))
    ag_target_dense = float(ag_cfg.get("target_dense", 0.1))
    ag_samples = int(ag_cfg.get("samples", 10000))

    # optional filters
    steps_allow = set(cfg.get("steps", []) or [])
    trainers_allow = set(cfg.get("trainers", []) or [])
    sparsity_allow = set(cfg.get("sparsity_levels", []) or [])

    # load index
    index_path = os.path.join(output_dir, "checkpoint_index.parquet")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"checkpoint index not found: {index_path}")

    df = pd.read_parquet(index_path)
    if df is None or len(df) == 0:
        print("[warn] empty checkpoint index; nothing to evaluate.")
        return

    # normalize columns
    for col in ("variant", "trainer", "step", "sparsity", "path"):
        if col not in df.columns:
            df[col] = None
    # cast
    if df["trainer"].dtype == object:
        df["trainer"] = pd.to_numeric(df["trainer"], errors="coerce").fillna(-1).astype(int)
    if df["step"].dtype == object:
        df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(-1).astype(int)
    if df["sparsity"].dtype == object:
        # allow None
        mask = df["sparsity"].notnull()
        df.loc[mask, "sparsity"] = pd.to_numeric(df.loc[mask, "sparsity"], errors="coerce")
        df["sparsity"] = df["sparsity"].fillna(-1).astype(int)

    # apply filters if non-empty
    def _allow(val, allowed: set):
        return True if not allowed else (val in allowed)

    rows = []
    pbar = tqdm(total=len(df), desc=f"eval checkpoints")
    for _, row in df.iterrows():
        pbar.update(1)
        variant = str(row["variant"] or "standard")
        trainer = int(row["trainer"])
        step = int(row["step"])
        sparsity = int(row["sparsity"]) if not pd.isna(row["sparsity"]) else -1
        path = str(row["path"])

        if not os.path.exists(path):
            continue
        if not _allow(trainer, trainers_allow):  # if filters provided
            continue
        if not _allow(step, steps_allow):
            continue
        # if sparsity not parsed (-1), do not filter on sparsity
        if sparsity != -1 and not _allow(sparsity, sparsity_allow):
            continue

        # load weights
        try:
            raw = _load_pt_or_safetensors(path)
            W_enc_raw, W_dec_raw, b_enc = _extract_sae_weights(raw)
        except Exception as e:
            print(f"[warn] failed to read weights @ {path}: {e}")
            continue

        # orient
        try:
            W_enc, D = _canon_orientations(W_enc_raw, W_dec_raw, d_in)
        except Exception as e:
            print(f"[warn] orientation error @ {path}: {e}")
            continue

        n_lat = W_enc.shape[1]

        # amortized codes on a manageable subset for metrics (corr_sub)
        n_use = min(corr_sub, Ntokens)
        X_sub = X[:n_use]  # [n_use, d_in]
        # guessed K for topk
        guessed_k = None
        for key in ("k", "topk_k", "K", "k_actives"):
            if key in raw and raw[key] is not None:
                try:
                    guessed_k = int(_np(raw[key]).reshape(()))
                    break
                except Exception:
                    pass
        Z = encode_amortized(X_sub, W_enc, b_enc, variant, target_dense=ag_target_dense, guessed_k=guessed_k)
        X_hat = Z @ D.T  # recon for NMSE

        # metrics row meta
        m_row = dict(
            variant=variant,
            trainer=trainer,
            step=step,
            sparsity=sparsity,
            path=path,
            d_in=d_in,
            n_lat=n_lat,
            n_tokens=n_use,
        )

        # NMSE (uses Z, not ReLU)
        if normalize_mse:
            m_row["NMSE"] = nmse(X_sub, X_hat)
        else:
            m_row["MSE"] = float(np.mean(np.sum((X_sub - X_hat) ** 2, axis=1)))

        # ---- Unified metrics on Z_relu ----
        Z_relu = np.maximum(Z, 0.0).astype(np.float32, copy=False)   # <-- FIX: remove non-existent dtype kw

        # dead & dense
        dead, dense = dead_and_dense_rates(Z_relu, dense_thresholds=dense_thresholds, dead_thr=dead_thr)
        m_row["dead_rate"] = dead
        for kname, val in dense.items():
            m_row[kname] = val

        # labels
        y = pseudo_labels_from_norm(X_sub)

        # k-sparse probe (corr-based feature order, standardized features, balanced LR)
        ks_sorted = tuple(k for k in sorted(set(probe_ks)) if isinstance(k, (int, np.integer)) and k >= 1)
        f1s, order = k_sparse_probe_corr(Z_relu, y, ks=ks_sorted if ks_sorted else (1,2))
        for kname, val in f1s.items():
            m_row[kname] = val

        # Absorption (corr-based, global main & aux)
        m_row["absorption"] = float("nan")
        try:
            m_row["absorption"] = absorption_from_corr(Z_relu, y, order, topk=absorption_topk)
        except Exception:
            pass

        # Amortization gap (optional; uses amortized Z on X_gap subset)
        if ag_enabled:
            n_gap = min(ag_samples, Ntokens)
            X_gap = X[:n_gap]
            Z_gap = encode_amortized(X_gap, W_enc, b_enc, variant, target_dense=ag_target_dense, guessed_k=guessed_k)
            used_lambda = None
            if ag_lambda is not None:
                used_lambda = float(ag_lambda)
            elif ag_calib:
                n_cal = min(max(2000, n_gap // 5), n_gap)
                used_lambda = calibrate_lambda(X_gap[:n_cal], D, target_dense=ag_target_dense, steps=ag_steps)
            else:
                used_lambda = 1e-3
            try:
                gap_mean, gap_median = amortization_gap(X_gap, Z_gap, D, lmbd=used_lambda, steps=ag_steps)
            except Exception as e:
                print(f"[warn] amortization gap failed @ {path}: {e}")
                gap_mean = float("nan"); gap_median = float("nan")
            m_row["amort_gap_mean"] = gap_mean
            m_row["amort_gap_median"] = gap_median
            m_row["amort_lambda"] = used_lambda

        rows.append(m_row)

    pbar.close()

    if not rows:
        print("[exp5] no rows evaluated (after loading/orientation/filters).")
        return

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "metrics.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"[exp5] done -> {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
