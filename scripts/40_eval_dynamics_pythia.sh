#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json, os, glob, numpy as np
from amortization_src.sae_io import load_sae_state
from amortization_src.amortization import ista_lasso, calibrate_lambda_by_dense_target

H = np.load("outputs/agnews_L8/H_last_token.npy")  # precomputed by 10_collect_agnews.sh
rows = []
for ck in sorted(glob.glob("ckpts/**/*.pt", recursive=True)):
    st = load_sae_state(ck)
    D = st["W_dec"].cpu().numpy().astype("float32")
    gate = None
    for k in ["gate","gates","gate_vec"]:
        if k in st: gate = st[k].cpu().numpy().ravel().astype("float32")
    if gate is not None and gate.shape[0]==D.shape[1]:
        D = D * gate.reshape(1,-1)

    lam = calibrate_lambda_by_dense_target(H, D, target_dense=0.10, steps=60)
    Z = ista_lasso(H, D, lam=lam, steps=200)
    nmse = float(np.mean(np.sum((H - Z @ D.T)**2, axis=1)) / np.mean(np.sum(H**2, axis=1)))
    dens = float((Z>0).mean())
    rows.append({"ckpt": ck, "NMSE": nmse, "density": dens})

os.makedirs("outputs", exist_ok=True)
with open("outputs/dynamics_summary.json","w") as f: json.dump(rows, f, indent=2)
print(f"[OK] wrote outputs/dynamics_summary.json with {len(rows)} rows")
PY
