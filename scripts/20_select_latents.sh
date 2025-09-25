#!/usr/bin/env bash
set -euo pipefail

CKPT="$1"                # path/to/your_sae_checkpoint.pt
MODE="${2:-semi}"        # amortized | semi | unamortized | topk_omp
TOPM="${3:-100}"
CLASS_ID="${4:-0}"

python -m amortization_src.select_latents_cli   --sae_ckpt "${CKPT}"   --encode_mode "${MODE}"   --class_id ${CLASS_ID}   --n_samples 5000   --topM ${TOPM}   --layer_idx 8   --out "outputs/latents_cls${CLASS_ID}_${MODE}_top${TOPM}.json"
