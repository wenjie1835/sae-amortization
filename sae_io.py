# src/exp5/sae_io.py
import os, torch

def _first_exist(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

def _strip_prefix(d, prefixes=("module.", "auto_encoder.", "model.", "ae.", "sae.")):
    out = {}
    for k,v in d.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_sae_state(path: str):
    """
    Return unified state:
      state["W_dec"] : [d, m] torch.float32, cpu
      (optional) state["W_enc"] : [m, d]
      (optional) state["b_enc"] : [m]
     (optional) state["b_dec"] : [d] / scaler etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(path, device="cpu")
    else:  # .pt / other
        raw = torch.load(path, map_location="cpu")

    # Extract state_dict
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            sd = raw["model_state_dict"]
        else:
            sd = raw
    else:
        # If it was a saved nn.Module
        try:
            sd = raw.state_dict()
        except Exception:
            raise ValueError(f"Unrecognized checkpoint structure for {path}")

    sd = _strip_prefix(sd)

    # Try alternative key names
    k_Wdec = _first_exist(sd, ["W_dec","W_D","decoder.weight","dict","dictionary","D"])
    if k_Wdec is None:
        raise KeyError(f"Cannot find decoder weight in {list(sd.keys())[:20]} ...")

    k_Wenc = _first_exist(sd, ["W_enc","encoder.weight","E","W_E"])
    k_benc = _first_exist(sd, ["b_enc","encoder.bias","bias_enc"])
    k_bdec = _first_exist(sd, ["b_dec","decoder.bias","bias_dec","b"])

    W_dec = sd[k_Wdec].float().cpu()
    state = {"W_dec": W_dec}

    if k_Wenc is not None:
        state["W_enc"] = sd[k_Wenc].float().cpu()
    if k_benc is not None:
        state["b_enc"] = sd[k_benc].float().cpu()
    if k_bdec is not None:
        state["b_dec"] = sd[k_bdec].float().cpu()

    # Optional: some trainings store scaler/threshold
    for extra in ["scaler","scale","threshold","beta","alpha"]:
        if extra in sd and torch.is_tensor(sd[extra]):
            state[extra] = sd[extra].float().cpu()

    # Basic shape sanity check
    d, m = state["W_dec"].shape
    if "W_enc" in state and state["W_enc"].shape != torch.Size([m, d]):
        # Some implementations store [d, m]; transpose automatically.
        if state["W_enc"].shape == torch.Size([d, m]):
            state["W_enc"] = state["W_enc"].T.contiguous()
        else:
            raise ValueError(f"Unexpected W_enc shape: {tuple(state['W_enc'].shape)}, expect ({m},{d})")

    return state
