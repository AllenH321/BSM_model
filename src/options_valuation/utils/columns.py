from __future__ import annotations

import pandas as pd


def pick_price_col(df: pd.DataFrame) -> str:
    """
    Prefer adj_close if exists; otherwise close.
    Handles multiple common spellings.
    """
    candidates = [
        "adj_close", "Adj Close", "adjclose", "adjusted_close",
        "close", "Close",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No known price column found. Columns={list(df.columns)}")
