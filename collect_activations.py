# coding: utf-8
"""
Collect layer activations for Gemma-2-2B with robust forward hook.
- Supports streaming dataset loading (datasets.load_dataset(..., streaming=True))
  to avoid downloading the full corpus; only the first `num_sequences` examples
  are consumed.
- Extracts activations from a chosen transformer block's output ("resid_post"-like),
  defaulting to layer index 12.
- Fixes the common forward-hook issue where module outputs can be tuple/dict/dataclass
  by extracting the first tensor-like field safely before .detach().
- Saves a single numpy array of shape [num_sequences, max_seq_len, hidden_dim]
  to outputs/acts_layer{layer_idx}.npy

Config keys expected in --config YAML:
dataset:
  name: monology/pile-uncopyrighted
  split: train
  text_field: text
  streaming: true
  num_sequences: 10000
  max_seq_len: 128
  batch_size: 8
model:
  name: google/gemma-2-2b
  layer_index: 12
  device: auto   # "auto" | "cuda" | "cpu"
seed: 123

Usage:
  python -m src.exp5.collect_activations --config config.yaml
"""

import os
import sys
import math
import argparse
import json
import random
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

try:
    import yaml
except Exception as e:
    yaml = None

try:
    from datasets import load_dataset
except Exception as e:
    load_dataset = None


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _first_tensor(x: Any) -> torch.Tensor:
    """
    Robustly fetch a tensor from various HF module outputs:
    - torch.Tensor
    - dataclass with .last_hidden_state
    - tuple/list of tensors (pick the first one)
    - dict containing common keys ('last_hidden_state', 'hidden_states', 'attn_output')
    """
    if torch.is_tensor(x):
        return x
    # dataclass-like
    if hasattr(x, "last_hidden_state") and torch.is_tensor(x.last_hidden_state):
        return x.last_hidden_state
    # tuple/list
    if isinstance(x, (tuple, list)):
        for t in x:
            if torch.is_tensor(t):
                return t
            if hasattr(t, "last_hidden_state") and torch.is_tensor(t.last_hidden_state):
                return t.last_hidden_state
    # dict
    if isinstance(x, dict):
        for key in ("last_hidden_state", "hidden_states", "attn_output"):
            if key in x and x[key] is not None:
                if torch.is_tensor(x[key]):
                    return x[key]
                if hasattr(x[key], "last_hidden_state") and torch.is_tensor(x[key].last_hidden_state):
                    return x[key].last_hidden_state
    raise TypeError(f"Hook output of type {type(x)} does not contain a tensor-like field.")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. `pip install pyyaml`")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(device_pref: str = "auto") -> str:
    if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    if device_pref == "mps" and torch.backends.mps.is_available():  # optional
        return "mps"
    return "cpu"


# ---------------------------
# Tokenization helpers
# ---------------------------

def batch_tokenize(texts: List[str], tokenizer, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


# ---------------------------
# Streaming dataset iterator
# ---------------------------

def stream_texts(ds_name: str, split: str, text_field: str, streaming: bool, num_sequences: int) -> Iterable[str]:
    """
    Yield up to `num_sequences` texts from the dataset.
    If streaming=True, use datasets streaming to avoid full download.
    Otherwise, fall back to split slicing (train[:N]).
    """
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. `pip install datasets`")

    if streaming:
        ds = load_dataset(ds_name, split=split, streaming=True)
        # Use islice to cut the stream to the first num_sequences
        for ex in islice(ds, num_sequences):
            txt = ex[text_field]
            if isinstance(txt, list):
                txt = " ".join(map(str, txt))
            yield str(txt)
    else:
        # non-streaming: slice the split to avoid full download
        sliced = f"{split}[:{num_sequences}]"
        ds = load_dataset(ds_name, split=sliced, streaming=False)
        for ex in ds:
            txt = ex[text_field]
            if isinstance(txt, list):
                txt = " ".join(map(str, txt))
            yield str(txt)



# src/exp5/collect_activations.py 
def _get_block(model, layer_idx):
    # Try common architectures in order: LLaMA/Gemma, GPT-NeoX (Pythia), GPT2/OPT, etc.
    cands = [
        ("model.layers", lambda m: m.model.layers[layer_idx]),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers[layer_idx]),
        ("transformer.h", lambda m: m.transformer.h[layer_idx]),
    ]
    last_err = None
    for _, getter in cands:
        try:
            return getter(model)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot locate block[{layer_idx}]: {last_err}")


# ---------------------------
# Main collection routine
# ---------------------------

def collect_activations(cfg: Dict[str, Any]):
    # Resolve config with defaults
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    ds_name = ds_cfg.get("name", "monology/pile-uncopyrighted")
    split = ds_cfg.get("split", "train")
    text_field = ds_cfg.get("text_field", "text")
    streaming = bool(ds_cfg.get("streaming", True))
    num_sequences = int(ds_cfg.get("num_sequences", 10000))
    max_seq_len = int(ds_cfg.get("max_seq_len", 128))
    batch_size = int(ds_cfg.get("batch_size", 8))

    model_name = model_cfg.get("name", "google/gemma-2-2b")
    layer_idx = int(model_cfg.get("layer_index", 12))
    device_pref = model_cfg.get("device", "auto")

    seed = int(cfg.get("seed", 123))

    out_dir = cfg.get("output_dir", "outputs")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"acts_layer{layer_idx}.npy")

    # Seed & device
    set_seed(seed)
    device = pick_device(device_pref)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # For causal LMs, set pad token to eos to enable padding
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()

    # reduce extra outputs to keep hook simpler
    if hasattr(model, "config"):
        model.config.output_attentions = False
        model.config.output_hidden_states = False
        model.config.use_cache = False

    model.to(device)

    # Locate the target block
    # Most llama/gemma-like models expose: model.model.layers[idx]
    try:
        target_module = model.model.layers[layer_idx]
    except Exception as e:
        raise RuntimeError(
            f"Cannot locate model.model.layers[{layer_idx}] on {model_name}: {e}"
        )

    # Prepare hook & buffer
    acts: List[torch.Tensor] = []

    def hook_fn(module, inp, out):
        t = _first_tensor(out)  # robustly get the tensor
        # Expect shape [bs, seq, hidden_dim]; move to CPU fp32
        acts.append(t.detach().to("cpu", dtype=torch.float32))

    handle = target_module.register_forward_hook(hook_fn)

    # Stream texts and run batches
    texts_iter = stream_texts(
        ds_name=ds_name,
        split=split,
        text_field=text_field,
        streaming=streaming,
        num_sequences=num_sequences,
    )

    # Batch the stream
    pending: List[str] = []
    pbar = tqdm(total=num_sequences, desc="collect acts (streaming)" if streaming else "collect acts")
    with torch.no_grad():
        for txt in texts_iter:
            pending.append(txt)
            if len(pending) >= batch_size:
                batch = batch_tokenize(pending, tokenizer, max_seq_len)
                for k in batch:
                    batch[k] = batch[k].to(device)
                # forward pass through full model (hook fires at layer_idx)
                _ = model(**batch)
                pbar.update(len(pending))
                pending = []
        # flush remainder
        if pending:
            batch = batch_tokenize(pending, tokenizer, max_seq_len)
            for k in batch:
                batch[k] = batch[k].to(device)
            _ = model(**batch)
            pbar.update(len(pending))
    pbar.close()

    # Remove hook
    handle.remove()

    if len(acts) == 0:
        raise RuntimeError("No activations were collected. Check hook location and data flow.")

    # Concatenate along batch dimension; expect list of [bs, seq, hid]
    # Stack -> [num_batches, bs, seq, hid] -> reshape to [N, seq, hid]
    arr = torch.cat(acts, dim=0).cpu().numpy()  # [total_bs, seq, hid]
    # If the last batch was smaller, arr.shape[0] can be >= num_sequences but not necessarily equal to it
    total_sequences = arr.shape[0]
    if total_sequences > num_sequences:
        arr = arr[:num_sequences]

    # Save numpy array
    np.save(out_path, arr)
    print(f"[OK] saved activations to: {out_path} with shape {arr.shape} "
          f"(num_sequences={arr.shape[0]}, seq_len={arr.shape[1]}, hidden={arr.shape[2]})")


# ---------------------------
# Entrypoint
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_yaml(args.config)
    collect_activations(cfg)


if __name__ == "__main__":
    main()
