
import numpy as np

def absorption_rate(latents: np.ndarray, labels: np.ndarray, order: np.ndarray, topk:int=5) -> float:
    """
    Approximate definition:
      - Primary latent idx0 = order[0]
      - ：Primary latent， topk ， absorption。
    """
    L = latents
    idx0 = int(order[0]); top = order[:topk]
    fire0 = (L[:, idx0] > 0)
    fire_any = (L[:, top] > 0).any(axis=1)
    pos = (labels == 1)
    absorbed = (~fire0) & fire_any & pos
    denom = pos.sum()
    return float(absorbed.sum() / max(denom, 1))
