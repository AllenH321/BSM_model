# src/options_valuation/pricing/bsm.py

from __future__ import annotations

import math
import pandas as pd
import numpy as np

from options_valuation.utils.volatility import annualize_sigma_from_horizon


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_price(
    S0: float,
    K: float,
    r_annual: float,
    sigma_annual: float,
    T_days: int,
    option_type: str = "call",
    trading_days: int = 252,
    q_annual: float = 0.0,
) -> float:
    """
    Black–Scholes–Merton price for EUROPEAN options.
    Uses annualized r, annualized sigma, and T in years (T_days/trading_days).
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T_days <= 0:
        raise ValueError("T_days must be positive.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    if sigma_annual < 0:
        raise ValueError("sigma_annual must be non-negative.")

    opt = option_type.lower().strip()
    if opt not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    T = T_days / float(trading_days)

    # Limiting case: sigma = 0
    if sigma_annual == 0.0:
        forward = S0 * math.exp((r_annual - q_annual) * T)
        disc = math.exp(-r_annual * T)
        if opt == "call":
            return float(disc * max(forward - K, 0.0))
        return float(disc * max(K - forward, 0.0))

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r_annual - q_annual + 0.5 * sigma_annual**2) * T) / (sigma_annual * sqrtT)
    d2 = d1 - sigma_annual * sqrtT

    disc_r = math.exp(-r_annual * T)
    disc_q = math.exp(-q_annual * T)

    if opt == "call":
        return float(S0 * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2))
    else:
        return float(K * disc_r * _norm_cdf(-d2) - S0 * disc_q * _norm_cdf(-d1))


def price_bsm_grid(
    ticker: str,
    S0: float,
    r_annual: float,
    K_grid,
    vol_table: pd.DataFrame,
    models=("garch", "hist_rolling"),
    horizons=range(1, 6),
    trading_days: int = 252,
    q_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Price BSM call/put over a strike grid and horizons for multiple volatility models.

    Expected vol_table columns:
      - ticker (str)
      - model (str)              e.g. 'garch', 'hist_rolling'
      - horizon_days (int)       e.g. 1..5
      - sigma (float)            horizon volatility over horizon_days (NOT annualized)

    Returns a DataFrame similar to CRR runner output:
      ticker, asof_S0, K, T_days, vol_model, sigma_h, sigma_ann, r_annual, q_annual, call_BSM, put_BSM
    """
    required = {"ticker", "model", "horizon_days", "sigma"}
    missing = required - set(vol_table.columns)
    if missing:
        raise ValueError(f"vol_table missing required columns: {sorted(missing)}")

    # defensive copy + types
    vt = vol_table.copy()
    vt["horizon_days"] = vt["horizon_days"].astype(int)
    vt["sigma"] = vt["sigma"].astype(float)

    results = []
    for model_choice in models:
        sub = vt[(vt["ticker"] == ticker) & (vt["model"] == model_choice)].copy()
        if sub.empty:
            raise ValueError(f"No volatility rows for ticker={ticker}, model={model_choice}")

        for T_days in horizons:
            row = sub[sub["horizon_days"] == int(T_days)]
            if row.empty:
                raise ValueError(f"Missing sigma for ticker={ticker}, model={model_choice}, horizon_days={T_days}")

            sigma_h = float(row["sigma"].iloc[0])
            sigma_ann = float(annualize_sigma_from_horizon(sigma_h, int(T_days), trading_days=trading_days))

            for K in K_grid:
                K = float(K)
                call = bsm_price(
                    S0=S0, K=K, r_annual=r_annual, sigma_annual=sigma_ann,
                    T_days=int(T_days), option_type="call",
                    trading_days=trading_days, q_annual=q_annual
                )
                put = bsm_price(
                    S0=S0, K=K, r_annual=r_annual, sigma_annual=sigma_ann,
                    T_days=int(T_days), option_type="put",
                    trading_days=trading_days, q_annual=q_annual
                )

                results.append({
                    "ticker": ticker,
                    "asof_S0": float(S0),
                    "K": K,
                    "T_days": int(T_days),
                    "vol_model": model_choice,
                    "sigma_h": sigma_h,
                    "sigma_ann": sigma_ann,
                    "r_annual": float(r_annual),
                    "q_annual": float(q_annual),
                    "call_BSM": call,
                    "put_BSM": put,
                })

    return (
        pd.DataFrame(results)
        .sort_values(["vol_model", "T_days", "K"])
        .reset_index(drop=True)
    )
