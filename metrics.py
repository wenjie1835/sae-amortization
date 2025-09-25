
import numpy as np

def mse(x: np.ndarray, xhat: np.ndarray, normalize: bool=True) -> float:
    err = ((x - xhat) ** 2).mean()
    if not normalize: return float(err)
    var = (x ** 2).mean()
    return float(err / (var + 1e-9))

def dead_dense_rates(act_rate: np.ndarray, dead_th: float=1e-6, dense_ths=(0.1,0.2)):
    dead = (act_rate <= dead_th).mean()
    dens = {th: (act_rate >= th).mean() for th in dense_ths}
    return float(dead), {float(k): float(v) for k,v in dens.items()}
