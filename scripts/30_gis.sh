#!/usr/bin/env bash
set -euo pipefail

CKPT="$1"                     # path/to/your_sae_checkpoint.pt
LAT_IDXES="$2"                # e.g., "123,456,789"
MODE="${3:-unamortized}"      # amortized | semi | unamortized
KL="${4:-0.33}"               # 0.10 | 0.33 | 1.00

python -m amortization_src.intervention_score   --model EleutherAI/pythia-160m-deduped   --sae_ckpt "${CKPT}"   --lat_idxes "${LAT_IDXES}"   --kl ${KL}   --encode_mode "${MODE}"   --layer_idx 8   --device cuda
