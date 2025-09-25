# SAE Amortization & Interventions (Pythia-160M Layer 8)

This repository contains a lightweight, **reproducible** pipeline to train/evaluate Sparse Autoencoders (SAEs) and to run **downstream interventions**:

- **TPP (Targeted Probe Perturbation)** style latent selection and ablations
- **GIS (Generative Intervention Scoring)** with KL-calibrated intervention strength

It is tailored to **EleutherAI/pythia-160m-deduped**, `resid_post`, **layer=8** by default and supports four SAE families:
**TopK**, **Standard**, **JumpReLU**, and **Gated**.

## What’s included

- Robust activation collection with architecture autodiscovery (LLaMA/Gemma, GPT‑NeoX/Pythia, GPT‑2/OPT).
- Unified latent encoding:
  - `amortized`: encoder forward (if available)
  - `semi`: amortized (or ALISTA one-step) init + a few ISTA steps
  - `unamortized`: ISTA from zero
  - `topk_omp`: OMP-like k-sparse coding (optional baseline)
- Gated-SAE **bugfix**: we always fold the learned **gate vector** into the decoder to obtain an **effective dictionary** for encoding/decoding and interventions.
- Intervention **RMS-normalization of Δ** so that the multiplier **α** is comparable across SAE variants.
- CLI helpers for latent selection (`select_latents_cli.py`) and GIS (`intervention_score.py`).

## Install

```bash
pip install -e .
# or:
pip install -r requirements.txt
```

> Requires Python 3.10+ and a recent PyTorch build with CUDA if you use GPU.

## Quick start

1) **Collect activations** (AG News example):
```bash
python -m amortization_src.collect_activations --config configs/collect_agnews.yaml
```

2) **Select latents for a target class (TPP-style)**:
```bash
python -m amortization_src.select_latents_cli \
  --sae_ckpt /path/to/sae.pt \
  --encode_mode semi \
  --class_id 0 \
  --n_samples 5000 \
  --topM 100 \
  --out outputs/latents_agnews_cls0.json
```

3) **Run GIS with KL calibration**:
```bash
python -m amortization_src.intervention_score \
  --sae_ckpt /path/to/sae.pt \
  --lat_idxes 123,456,789 \
  --kl 0.33 \
  --encode_mode unamortized
```

## Key design choices / fixes

- **Autodetect block**: `collect_activations._get_block` tries common module paths and registers a forward hook robustly.
- **Gated-SAE**: we compute an **effective decoder** `D_eff = D * diag(gate)` and use it **everywhere** (encoding, λ calibration, and Δ construction).
- **Alpha calibration stability**: we RMS-normalize the per-batch Δ so α behaves similarly across models.
- **Tokenizer**: we guarantee a valid `pad_token`. If missing, EOS is cloned as PAD.

## Project layout

```
amortization_src/
  absorption.py
  amortization.py
  collect_activations.py
  encode_latents.py
  geometry.py
  index_checkpoints.py
  intervention_score.py
  metrics.py
  probes.py
  run_exp5.py
  sae_io.py
  select_latents_cli.py
  tpp_shift.py
  utils.py
```

## Reproducibility tips

- Use `encode_mode` consistently when comparing SAE variants.
- Use the same **latent set S** across amortization modes for apples-to-apples comparisons, then optionally report each mode’s own top‑M as a supplement.
- Cache activations and codes (`.npy/.npz`) to make runs deterministic and fast.

## License

MIT
