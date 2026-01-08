import numpy as np

def make_strike_grid(
    S0: float,
    offsets=(-2, -1, 0, 1, 2),
    step: float = 1.0,
    round_to: float = 0.01
) -> np.ndarray:
    """
    Create strike grid K = S0 + offset*step for given offsets.
    Rounds to round_to (e.g. 0.01 for cents) and ensures K > 0.
    """
    Ks = [S0 + o * step for o in offsets]
    Ks = [max(k, round_to) for k in Ks]
    Ks = [round(k / round_to) * round_to for k in Ks]
    
    return np.array(sorted(set(Ks)), dtype=float)
