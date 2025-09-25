#!/usr/bin/env bash
set -euo pipefail

MODEL="EleutherAI/pythia-160m-deduped"
LAYER=8
OUTDIR="outputs/agnews_L${LAYER}"
mkdir -p "${OUTDIR}"

python -m amortization_src.collect_activations   --model_name "${MODEL}"   --layer_index ${LAYER}   --dataset.name ag_news   --dataset.split train   --dataset.streaming true   --dataset.num_sequences 50000   --dataset.max_seq_len 128   --output_dir "${OUTDIR}"
