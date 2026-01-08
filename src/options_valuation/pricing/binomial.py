import numpy as np
import pandas as pd
from options_valuation.utils.volatility import annualize_sigma_from_horizon

def crr_binomial_price(
    S0: float,
    K: float,
    r_annual: float,
    sigma_annual: float,
    T_days: int,
    option_type: str = "call",
    steps: int | None = None,
    trading_days: int = 252
) -> float:
    """
    Cox-Ross-Rubinstein binomial tree for AMERICAN options.
    Uses annualized r and annualized sigma.
    Early exercise is allowed at every node.
    """
    if T_days <= 0:
        raise ValueError("T_days must be positive.")
    if steps is None:
        steps = int(T_days)
    if steps <= 0:
        raise ValueError("steps must be positive.")

    T = T_days / float(trading_days)
    dt = T / steps

    # CRR parameters
    u = np.exp(sigma_annual * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r_annual * dt)
    p = (np.exp(r_annual * dt) - d) / (u - d)
    p = float(np.clip(p, 0.0, 1.0))

    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # Terminal stock prices (time = steps)
    j = np.arange(steps + 1)
    ST = S0 * (u ** j) * (d ** (steps - j))

    # Terminal payoff
    if opt == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction with early exercise check
    for t in range(steps - 1, -1, -1):
        # continuation value
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

        # underlying prices at time t (length t+1)
        j = np.arange(t + 1)
        ST = S0 * (u ** j) * (d ** (t - j))

        # intrinsic value at time t
        if opt == "call":
            intrinsic = np.maximum(ST - K, 0.0)
        else:
            intrinsic = np.maximum(K - ST, 0.0)

        # American: choose max(continue, exercise)
        V = np.maximum(V, intrinsic)

    return float(V[0])


def price_crr_grid(
    ticker: str,
    S0: float,
    r_annual: float,
    K_grid,
    vol_table: pd.DataFrame,
    models=("garch", "hist_rolling"),
    horizons=range(1, 6)
) -> pd.DataFrame:
    """
    Price call/put over strike grid and horizons for multiple volatility models.
    vol_table must contain columns: ticker, model, horizon_days, sigma (horizon sigma).
    """
    results = []
    for model_choice in models:
        sub = vol_table[(vol_table["ticker"] == ticker) & (vol_table["model"] == model_choice)].copy()
        if sub.empty:
            raise ValueError(f"No volatility rows for {ticker} model={model_choice}")

        for T_days in horizons:
            row = sub[sub["horizon_days"] == T_days]
            if row.empty:
                raise ValueError(f"Missing sigma for {ticker} model={model_choice} horizon_days={T_days}")

            sigma_h = float(row["sigma"].iloc[0])
            sigma_ann = annualize_sigma_from_horizon(sigma_h, T_days)

            for K in K_grid:
                call = crr_binomial_price(S0, float(K), r_annual, sigma_ann, T_days, "call", steps=T_days)
                put  = crr_binomial_price(S0, float(K), r_annual, sigma_ann, T_days, "put",  steps=T_days)

                results.append({
                    "ticker": ticker,
                    "asof_S0": float(S0),
                    "K": float(K),
                    "T_days": int(T_days),
                    "vol_model": model_choice,
                    "sigma_h": sigma_h,
                    "sigma_ann": sigma_ann,
                    "r_annual": float(r_annual),
                    "call_CRR": call,
                    "put_CRR": put
                })

    return pd.DataFrame(results).sort_values(["vol_model", "T_days", "K"]).reset_index(drop=True)



