from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class SpotInfo:
    ticker: str
    S0: float
    asof_date: object  # datetime.date
    price_col: str
    file_path: Path


def get_latest_spot_info(
    project_root: Path,
    ticker: str | None = None,
    prefer_adj_close: bool = True,
) -> SpotInfo:
    """
    Returns the latest spot price info from data/raw.

    If the ticker is None, infer it from the most recently modified '*_daily_*.csv' file.
    If a ticker is provided, choose the most recent file matching '{ticker}_*_daily_*.csv'.
    """
    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

    pattern = f"{ticker}_*_daily_*.csv" if ticker else "*_daily_*.csv"

    files = sorted(
        raw_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {raw_dir}")

    latest_file = files[0]

    # Infer ticker if not provided
    if ticker is None:
        # assumes filename like AAPL_yfinance_daily_2026-01-08_....
        ticker = latest_file.name.split("_")[0]

    df = pd.read_csv(latest_file)

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {latest_file.name}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    price_col = "adj_close" if (prefer_adj_close and "adj_close" in df.columns) else "close"
    if price_col not in df.columns:
        raise ValueError(f"'{price_col}' column not found in {latest_file.name}")

    S0 = float(df[price_col].iloc[-1])
    asof_date = df["date"].iloc[-1].date()

    return SpotInfo(
        ticker=ticker,
        S0=S0,
        asof_date=asof_date,
        price_col=price_col,
        file_path=latest_file,
    )
