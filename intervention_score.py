# src/exp5/intervention_score.py
from __future__ import annotations
from typing import List, Callable
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

# Support two import styles (project root or src/exp5).
try:
    from .encode_latents import encode_latents
    from .sae_io import load_sae_state
    from .collect_activations import _get_block
except ImportError:
    from encode_latents import encode_latents
    from sae_io import load_sae_state
    from collect_activations import _get_block


@torch.no_grad()
def avg_token_kl(p_logits: Tensor, q_logits: Tensor) -> float:
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    return torch.sum(torch.exp(p) * (p - q), dim=-1).mean().item()


def _ensure_tok_model(model_name: str, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added_pad = False
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            added_pad = True
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    if added_pad and mdl.get_input_embeddings().num_embeddings < len(tok):
        mdl.resize_token_embeddings(len(tok))
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id
    mdl = mdl.to(device).eval()
    return tok, mdl


def _forward_with_hook(
    mdl: AutoModelForCausalLM,
    ids: dict,
    hook_fn: Callable[[Tensor], Tensor],
    layer_idx: int = 8,
):
    """
    Register a hook at the block output of the given layer_idx.
    - If the module returns a tuple, we only replace the first item (hidden_states); keep others intact.
    - The hook_fn input/output is [B, L, d].
    """
    blk = _get_block(mdl, layer_idx)

    def wrapper(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            t = out[0]
            t = t.to(torch.float32)
            new_t = hook_fn(t)
            # Preserve structure: only replace the first item.
            if isinstance(out, tuple):
                return (new_t,) + tuple(out[1:])
            else:
                out[0] = new_t
                return out
        else:
            t = out.to(torch.float32)
            new_t = hook_fn(t)
            return new_t

    handle = blk.register_forward_hook(wrapper)
    out = mdl(**ids, use_cache=False)  # Disable cache to avoid output structure changes.
    handle.remove()
    return out


def _make_delta_bld_from_last_token(
    t_bld: Tensor,              # [B, L, d]
    last_idx: Tensor,           # [B]
    sae_ckpt: str,
    encode_mode: str,
    Wdec_dm: Tensor,            # [d, m]
    lat_idxes: List[int],
    alpha: float,
) -> Tensor:
    """
    Compute codes at each sample's last non-pad token and build Δ; then broadcast to [B,L,d].
    """
    device = t_bld.device
    B, L, d = t_bld.shape

    # 1) Get the hidden at the last non-pad position: [B, d]
    h_last_bd = t_bld[torch.arange(B, device=device), last_idx]  # [B, d]

    # 2) SAE encoding
    H_np = h_last_bd.detach().to("cpu").numpy().astype(np.float32, copy=False)
    Z_np = encode_latents(
        X=H_np,
        sae_ckpt_path=sae_ckpt,
        mode=encode_mode,        # amortized / semi / unamortized / topk_omp
        ista_steps=200,
        semi_steps=10,
        target_dense=0.10,
        k_for_topk=20,
    )
    Z_bm = torch.from_numpy(Z_np).to(device=device, dtype=torch.float32)  # [B, m]

    # 3) Select only the target latents to intervene
    S_idx = torch.as_tensor(lat_idxes, device=device, dtype=torch.long)   # [S]
    Z_bs = Z_bm[:, S_idx]                                                 # [B, S]
    D_ds = Wdec_dm[:, S_idx]                                              # [d, S]

    # 4) Δ[B,d] = -(Z_bs @ D_ds^T) -> expand to [B,L,d]
    delta_bd = -(Z_bs @ D_ds.T)                                           # [B, d]
# RMS-normalize delta to decouple alpha from absolute scale differences
rms = torch.sqrt((delta_bd ** 2).mean()) + 1e-8
delta_bd = delta_bd / rms
delta_bld = delta_bd.unsqueeze(1).expand(B, L, d).to(dtype=t_bld.dtype)
return alpha * delta_bld


def intervention_score(
    model_name: str,
    sae_ckpt: str,
    lat_idxes: List[int],
    prompts: List[str],
    target_KL: float = 0.33,
    encode_mode: str = "unamortized",
    layer_idx: int = 8,
    device: str = "cuda",
    max_len: int = 128,
) -> dict:
    """
    Perform one KL-calibrated zero-ablation intervention on given prompts:
      - - First, compute clean logits;
      - - At layer_idx, subtract (zero) the chosen latents; binary-search α to match KL≈target_KL;
      - - Return {"alpha","KL",...}. If you implemented a scorer LLM, also report "score_S".
    """
    assert len(lat_idxes) > 0, "lat_idxes is empty; please provide SAE latent indices to intervene."

    tok, mdl = _ensure_tok_model(model_name, device=device)

    # SAE dictionary (loaded once).
    state = load_sae_state(sae_ckpt)
    Wdec_dm = state["W_dec"].to(device=device, dtype=torch.float32)  # [d, m]
# Fold gate vector (if present) into decoder to obtain effective dictionary
gate = None
for k in ["gate", "gates", "gate_vec"]:
    if k in state:
        gate = state[k].to(device=device, dtype=torch.float32).view(-1)
        break
if gate is not None and gate.numel() == Wdec_dm.shape[1]:
    Wdec_dm = Wdec_dm * gate.view(1, -1)

    #  logits
    with torch.no_grad():
        ids = tok(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len
        ).to(device)
        out_clean = mdl(**ids, use_cache=False)
        clean_logits_bv = out_clean.logits[:, -1, :].to(torch.float32)  # [B,V]

    # Last non-pad position for each sample.
    attn_mask = ids["attention_mask"]                      # [B, L]
    last_idx = (attn_mask.sum(dim=1) - 1).to(device)       # [B]

    # Binary search α.
    lo, hi = 0.0, 10.0
    best_alpha, best_kl = None, float("inf")

    for _ in range(12):
        mid = (lo + hi) / 2.0

        def hook_fn(t_bld: Tensor) -> Tensor:
            # t_bld: [B, L, d]
            delta = _make_delta_bld_from_last_token(
                t_bld=t_bld,
                last_idx=last_idx,
                sae_ckpt=sae_ckpt,
                encode_mode=encode_mode,
                Wdec_dm=Wdec_dm,
                lat_idxes=lat_idxes,
                alpha=mid,
            )  # [B, L, d]
            return t_bld + delta

        out_inter = _forward_with_hook(mdl, ids, hook_fn, layer_idx=layer_idx)
        inter_logits_bv = out_inter.logits[:, -1, :].to(torch.float32)
        kl = avg_token_kl(clean_logits_bv, inter_logits_bv)

        if abs(kl - target_KL) < abs(best_kl - target_KL):
            best_alpha, best_kl = mid, kl

        if kl < target_KL:
            lo = mid
        else:
            hi = mid

        if abs(kl - target_KL) / max(target_KL, 1e-6) < 0.10:
            best_alpha, best_kl = mid, kl
            break

    return {
        "alpha": float(best_alpha if best_alpha is not None else 0.0),
        "KL": float(best_kl if best_alpha is not None else 0.0),
        "n_prompts": int(len(prompts)),
        "encode_mode": encode_mode,
        "layer_idx": int(layer_idx),
    }


# Optional: CLI entry point for quick testing.
if __name__ == "__main__":
    import argparse, os, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--sae_ckpt", required=True)
    parser.add_argument("--lat_idxes", type=str, required=True)  # e.g. "123,456,789"
    parser.add_argument("--kl", type=float, default=0.33)
    parser.add_argument("--encode_mode", default="unamortized",
                        choices=["amortized", "semi", "unamortized", "topk_omp"])
    parser.add_argument("--layer_idx", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    lat_idxes = [int(x) for x in args.lat_idxes.split(",") if x.strip()]
    prompts = [
        "She told her sister about the plan.",
        "Compute 7 * 6 then add 5.",
        "The market rallied as investors sought safe havens."
    ]
    res = intervention_score(
        model_name=args.model,
        sae_ckpt=args.sae_ckpt,
        lat_idxes=lat_idxes,
        prompts=prompts,
        target_KL=args.kl,
        encode_mode=args.encode_mode,
        layer_idx=args.layer_idx,
        device=args.device,
    )
    os.makedirs("results/intervention", exist_ok=True)
    out_path = f"results/intervention/{os.path.basename(args.sae_ckpt)}.{args.encode_mode}.KL{args.kl}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[OK] saved -> {out_path}")
