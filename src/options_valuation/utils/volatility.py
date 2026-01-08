import numpy as np

def annualize_sigma_from_horizon(sigma_h: float, horizon_days: int, trading_days: int = 252) -> float:
    """
    Convert horizon volatility (sigma over horizon_days) to annualized sigma.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
        
    return float(sigma_h) / np.sqrt(horizon_days / trading_days)
