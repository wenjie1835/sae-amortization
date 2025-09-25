# THE PRICE OF AMORTIZED INFERENCE IN SPARSE AUTOENCODERS

> **Paper abstract**
> Polysemy has long been a major challenge in Mechanistic Interpretability (MI), with Sparse Autoencoders (SAEs) emerging as a promising solution. SAEs employ a shared encoder to map inputs to sparse codes, thereby amortizing inference costs across all instances. However, this parameter-sharing paradigm inherently conflicts with the MI community’s emphasis on instance-level optimality. We study the “price” of amortized inference by analyzing training dynamics and comparing fully-, semi-, and non-amortized inference. We find that many pathologies (dead features, splitting, absorption, density issues) are driven less by the presence of an encoder and more by the objectives and optimization dynamics. Moving beyond fully amortized inference (e.g., with semi- or non-amortized codes) alleviates several limitations and improves downstream controllability.&#x20;

---

## Repository scope

This repo provides code and scripts to reproduce the **main experiments** of the paper:

* **§4.3 Training Dynamics** — how dead/dense latents, splitting, absorption, NMSE, and the amortization gap evolve across sparsity settings and training steps. We follow the SAEBench setup (Gemma-2-2B, layer 12) and estimate non-amortized references with ISTA (≈200 steps).&#x20;
* **§5 Do pathologies really stem from amortized inference?** — a controlled comparison on **Pythia-160M-deduped**, layer 8, across **Standard / JumpReLU / Gated / Top-K** SAEs and **fully / semi / non-amortized** inference, with matched density and consistent evaluation metrics.&#x20;

> Note: Concept-perturbation / intervention tasks (TPP, GIS) are included only as **appendix-level** examples and minimal scripts. The core emphasis is the **training-dynamics analysis** and **§5 main comparison**.

---

## Installation

```bash
# (Optional) create an environment
python -m venv .venv
source .venv/bin/activate

# install this repo
pip install -e .
# or
pip install -r requirements.txt
```

Python ≥3.10 and a recent PyTorch+CUDA build are recommended for GPU runs.

---

## Data & models

* **Models**

  * Gemma-2-2B (SAEBench setup) for §4.3 training-dynamics replication.&#x20;
  * EleutherAI/**pythia-160m-deduped**, **resid\_post**, **layer=8** for §5 main results.&#x20;

* **Corpora / evaluation**

  * For §4.3 we mirror SAEBench: train/evaluate SAE checkpoints across sparsity regimes and steps; metrics are defined in the paper’s evaluation section/appendix.&#x20;
  * For §5 we stream a small held-out slice to **calibrate λ** (match average density across inference modes) and compute metrics (NMSE, Dead/Dense, Splitting/Absorption, etc.).&#x20;

---

## Reproducing the main experiments

### 1) §4.3 — Training dynamics (Gemma-2-2B, L12)

This section studies how pathologies co-evolve during training and how the amortization gap (vs. per-sample optimum via ISTA) behaves. We follow the SAEBench training recipe and evaluate at multiple sparsity levels and checkpoints; see Sec. 4.3 + Appx. E/F/H for metrics and protocol.&#x20;

**A. Prepare/check SAE checkpoints**

* Train (or download) **Standard** and **Top-K** SAEs at **six sparsity settings** with **seven step checkpoints** each, as in the paper (Gemma-2-2B / layer 12).&#x20;

**B. Run evaluation over the trajectory**

```bash
# Example: iterate over checkpoints under ckpts/{variant}/{sparsity_id}/{step}.pt
python -m amortization_src.eval_dynamics \
  --model_name google/gemma-2-2b \
  --layer_index 12 \
  --ckpt_root ckpts/gemma2b_L12 \
  --out outputs/dynamics_gemma2b_L12.json \
  --ista_steps 200
```

This script:

* Loads each checkpoint’s decoder `D` (and gate, if present).
* Calibrates `λ` to match a target density, then computes **non-amortized** codes with **ISTA=200** as a per-sample reference.
* Logs NMSE, Dead/Dense rates, Splitting/Absorption proxies, and an **amortization gap** estimate (objective difference) per checkpoint.
  Interpretation aligns with Fig. 1–2 and the observations in §4.3 (e.g., sparsity often raises NMSE & dead-rate and doesn’t reliably fix dense latents; splitting/absorption act as compensatory mechanisms; lower global gap ≠ better monosemanticity).&#x20;

---

### 2) §5 — Main comparison across amortization modes (Pythia-160M, L8)

This section compares **fully / semi / non-amortized** inference under four SAE variants on **pythia-160m-deduped / resid\_post / layer=8**, with density-matched calibration and unified metrics (see §5.1–5.3).&#x20;

**A. Collect activations (cached for reuse)**

```bash
python -m amortization_src.collect_activations \
  --model_name EleutherAI/pythia-160m-deduped \
  --layer_index 8 \
  --dataset.name ag_news \
  --dataset.split train \
  --dataset.streaming true \
  --dataset.num_sequences 50000 \
  --dataset.max_seq_len 128 \
  --output_dir outputs/pythia160m_L8
```

**B. Run the §5 evaluation sweep**

```bash
python -m amortization_src.eval_section5 \
  --model_name EleutherAI/pythia-160m-deduped \
  --layer_index 8 \
  --ckpts ckpts/TopK/pythia160m_L8.safetensors \
          ckpts/Gated/pythia160m_L8.safetensors \
          ckpts/JumpRelu/pythia160m_L8.safetensors \
          ckpts/Standard/pythia160m_L8.safetensors \
  --encode_modes amortized semi unamortized \
  --target_dense 0.10 \
  --ista_steps 200 \
  --out outputs/section5_pythia160m_L8.json
```

What this does (mirrors §5):

* Uses a **shared decoder** per checkpoint; folds **gate** into `D` for Gated SAEs; calibrates `λ` so that fully/semi/non-amortized runs have **matched average density** (fair comparison).&#x20;
* Computes NMSE, Dead/Dense@τ, Splitting/Absorption proxies, and aggregates per variant × mode.
* Produces summary CSV/JSON and plots comparable to Fig. 3 (Pythia-160M) in the paper.&#x20;

---

## (Appendix) Minimal scripts for TPP / GIS

These are **optional** appendix-level utilities; they are not the focus of this repository.

* **TPP**: trains linear probes, ranks latents by attribution, zero-ablates top-M; logs on-target drop / off-target leakage / selectivity.&#x20;
* **GIS**: applies **KL-calibrated** generative interventions and reports an **Intervention Score** using a scorer LM.&#x20;

> The paper treats these as supplementary evidence; please rely on §4.3 and §5 for the main takeaways.

---

## Tips & expected behavior

* **Amortization modes**:

  * *Fully*: encoder forward.
  * *Semi*: amortized (or ALISTA) init + a few ISTA steps.
  * *Non*: ISTA from zero (≈200 steps).
    Matching density is crucial for fair comparisons (we calibrate `λ` on a held-out slice).&#x20;

* **Gated SAEs**: always fold the learned **gate** into the decoder to build an **effective dictionary** for encoding and interventions; this avoids misleading scale effects when comparing modes.

* **Determinism**: cache activations and codes; set seeds where applicable.

---

## Citation

If you use this code or build upon our results, please cite the paper:
**“THE PRICE OF AMORTIZED INFERENCE IN SPARSE AUTOENCODERS.”** (ICLR 2026 under review).&#x20;

---

## License

MIT (see `LICENSE` in the repository).、
