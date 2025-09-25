# src/exp5/select_latents_cli.py
"""
Select top-M SAE latent indices for a target class (AG News) using the TPP-style score:
    score_j = <W_dec[:, j], w_probe> * (E[z_j | y=1] - E[z_j | y=0])

- Read a batch from AG News
- Collect last-token hidden states at resid_post (layer_idx) for Pythia-160m
- Train one-vs-rest linear probe to get w_probe
- Encode with the chosen mode (amortized / semi / unamortized / topk_omp) to get Z
- Compute the score and select the top-M latents
- Output JSON with indices and activation stats to verify non-degeneracy

Usage (example):
python -m src.exp5.select_latents_cli \
  --sae_ckpt "/root/.../GatedSAE_.../resid_post_layer_8/trainer_0/ae.pt" \
  --encode_mode semi \
  --class_id 0 \
  --topM 50 \
  --out results/latents/Gated/semi/class0.top50.json
"""

from __future__ import annotations
import os, json
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Support two import paths (project root or src/exp5).
try:
    from .encode_latents import encode_latents
    from .sae_io import load_sae_state
except ImportError:
    from encode_latents import encode_latents
    from sae_io import load_sae_state


# --------- helpers ---------
@torch.no_grad()
def _ensure_tok_model(model_name: str, device: str = "cuda"):
    """Ensure tokenizer/model has a pad_token and move model to device."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added_pad = False
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token     = tok.eos_token
            tok.pad_token_id  = tok.eos_token_id
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            added_pad = True

    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    if added_pad and mdl.get_input_embeddings().num_embeddings < len(tok):
        mdl.resize_token_embeddings(len(tok))
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id

    mdl = mdl.to(device).eval()
    return tok, mdl

@torch.no_grad()
def collect_hidden_last_token(model_name: str, texts: List[str], layer_idx: int = 8,
                              device: str = "cuda", batch_size: int = 32, max_len: int = 128) -> np.ndarray:
    """Batch-collect last-token hidden states at resid_post layer_idx; shape [N, d]."""
    tok, mdl = _ensure_tok_model(model_name, device)
    H_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        ids = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        out = mdl(**ids, output_hidden_states=True, use_cache=False)
        # hidden_states[0] are embeddings; resid_post at layer k is at hidden_states index k+1.
        h_layer = out.hidden_states[layer_idx + 1]            # [B, L, d]
        mask = ids["attention_mask"]                          # [B, L]
        last_idx = mask.sum(dim=1) - 1                        # [B]
        B = h_layer.size(0)
        H_last = h_layer[torch.arange(B, device=device), last_idx]  # [B, d]
        H_chunks.append(H_last.detach().to("cpu"))
    H = torch.cat(H_chunks, dim=0).numpy().astype(np.float32) # [N, d]
    return H


# --------- core selection ---------
def select_latents_for_class(
    model_name: str,
    sae_ckpt: str,
    encode_mode: str,
    cls: int,
    n_samples: int = 5000,
    topM: int = 50,
    layer_idx: int = 8,
    device: str = "cuda",
) -> Dict:
    # 1) Data
    ds = load_dataset("ag_news", split=f"train[:{n_samples}]")
    texts = ds["text"]; y = np.array(ds["label"])

    # 2) Last-token hidden states
    H = collect_hidden_last_token(model_name, texts, layer_idx=layer_idx, device=device)  # [N, d]

    # 3) One-vs-rest probe
    y_bin = (y == cls).astype(int)
    clf = make_pipeline(
        StandardScaler(with_mean=False),  # /
        LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=4)
    )
    clf.fit(H, y_bin)
    w_probe = clf.named_steps["logisticregression"].coef_.astype(np.float32).ravel()  # [d]

    # 4) Dictionary & encoding
    state = load_sae_state(sae_ckpt)
    D = state["W_dec"].cpu().numpy().astype(np.float32)                 # [d, m]
    Z = encode_latents(H, sae_ckpt, mode=encode_mode,
                       target_dense=0.10, ista_steps=200, semi_steps=10) # [N, m]

    # 5) Score and select top-M
    dot = (D.T @ w_probe)                                               # [m]
    dz  = Z[y_bin==1].mean(0) - Z[y_bin==0].mean(0)                     # [m]
    score = dot * dz
    idx = np.argsort(-np.abs(score))[:topM].tolist()

    # Diagnostics (activation rate etc.)
    pos_act  = float((Z[y_bin==1][:, idx] > 0).mean())
    neg_act  = float((Z[y_bin==0][:, idx] > 0).mean())
    pos_mean = float(Z[y_bin==1][:, idx].mean())
    neg_mean = float(Z[y_bin==0][:, idx].mean())

    return {
        "sae_ckpt": sae_ckpt,
        "encode_mode": encode_mode,
        "class_id": int(cls),
        "class_name": {0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}.get(int(cls), str(cls)),
        "topM": int(topM),
        "idx": idx,
        "stats": {
            "pos_active_frac": pos_act,
            "neg_active_frac": neg_act,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "dict_size": int(D.shape[1]),
            "d_model": int(D.shape[0]),
            "n_samples": int(n_samples),
        }
    }


# --------- CLI ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--sae_ckpt", required=True)
    parser.add_argument("--encode_mode", default="semi",
                        choices=["amortized","semi","unamortized","topk_omp"])
    parser.add_argument("--class_id", type=int, default=2, help="0=World,1=Sports,2=Business,3=Sci/Tech")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--topM", type=int, default=50)
    parser.add_argument("--layer_idx", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    res = select_latents_for_class(
        model_name=args.model,
        sae_ckpt=args.sae_ckpt,
        encode_mode=args.encode_mode,
        cls=args.class_id,
        n_samples=args.n_samples,
        topM=args.topM,
        layer_idx=args.layer_idx,
        device=args.device
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[OK] wrote -> {args.out}")
