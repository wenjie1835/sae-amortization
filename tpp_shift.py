# src/exp5/tpp_shift.py
import numpy as np, torch, datasets
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .encode_latents import encode_latents
from .sae_io import load_sae_state
from .collect_activations import _get_block

def collect_hidden_last_token(model_name: str, texts, layer_idx: int = 8, device="cuda",
                              batch_size: int = 32, max_len: int = 128):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # --- ： pad token ---
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            # ： [PAD]，
            tok.add_special_tokens({'pad_token': '[PAD]'})
    mdl = AutoModelForCausalLM.from_pretrained(model_name)

    #  [PAD]，
    if mdl.get_input_embeddings().num_embeddings < len(tok):
        mdl.resize_token_embeddings(len(tok))

    #  pad_token_id
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id

    mdl = mdl.to(device).eval()

    H_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            ids = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            out = mdl(**ids, output_hidden_states=True, use_cache=False)
            # hidden_states[0]  embeddings， k  resid_post  index = k+1
            h_layer = out.hidden_states[layer_idx + 1]            # [B, L, d]
            mask = ids["attention_mask"]                           # [B, L]
            last_idx = mask.sum(dim=1) - 1                         # pad
            B = h_layer.size(0)
            H_last = h_layer[torch.arange(B, device=device), last_idx]  # [B, d]
            H_chunks.append(H_last.cpu())
    H = torch.cat(H_chunks, dim=0).numpy()  # [N, d]
    return H

def select_latents(Wdec: np.ndarray, w_probe: np.ndarray, Z_pos: np.ndarray, Z_neg: np.ndarray, topM=100):
    # ：<D_j, w_probe> * (E[z_j|pos] - E[z_j|neg])
    delta = Z_pos.mean(0) - Z_neg.mean(0)          # [m]
    dot = (Wdec.T @ w_probe)                       # [m]
    score = dot * delta
    idx = np.argsort(-np.abs(score))[:topM]
    return idx

def ablate_latents(H: np.ndarray, Wdec: np.ndarray, Z: np.ndarray, idx):
    # h' = h + D(z' - z)， z'_S = 0
    D = Wdec
    delta = -(Z[:, idx] @ D[:, idx].T)             # [N, d]
    return H + delta

def tpp_on_agnews(model_name: str, sae_ckpt: str, encode_mode: str, layer_idx=8, topM=100, device="cuda"):
    ds = datasets.load_dataset("ag_news", split="train[:5000]")
    texts = ds["text"]; labels = np.array(ds["label"])
    H = collect_hidden_last_token(model_name, texts, layer_idx, device)  # [N, d]
    #  one-vs-rest 
    classes = sorted(set(labels))
    res = []
    state = load_sae_state(sae_ckpt)
    Wdec = state["W_dec"].cpu().numpy().astype(np.float32)  # [d, m]
    #  encode_mode  Z
    Z = encode_latents(H, sae_ckpt, mode=encode_mode, target_dense=0.10, ista_steps=200, semi_steps=10)
    for c in classes:
        y = (labels == c).astype(int)
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=4).fit(H, y)
        w = clf.coef_.astype(np.float32).ravel()           # [d]
        idx = select_latents(Wdec, w, Z[y==1], Z[y==0], topM=topM)
        base_pred = clf.predict(H)
        H_abl = ablate_latents(H, Wdec, Z, idx)
        pred_abl = clf.predict(H_abl)
        acc_base = accuracy_score(y, base_pred)
        acc_abl  = accuracy_score(y, pred_abl)
        res.append({"class": int(c), "acc_base": float(acc_base), "acc_after": float(acc_abl),
                    "drop": float(acc_base - acc_abl), "topM": int(topM),
                    "encode_mode": encode_mode, "sae_ckpt": sae_ckpt})
    return res

def shift_on_bias_in_bios(model_name: str, sae_ckpt: str, encode_mode: str, layer_idx=8, device="cuda"):
    """
    Standard procedure:
      1) Construct Biased vs Balanced splits (e.g., Profession×Gender long-tail).
      2) Train spurious probe and main probe,
      3) Use the attribution + zero-ablation pipeline to compute:
         S = (acc_after - acc_base) / (acc_skyline - acc_base)
       The skyline uses a probe trained only on spurious signals to define an upper bound.
    ， dict  S。
    """
    # Implement with your chosen professions; similar to tpp_on_agnews.
    raise NotImplementedError
