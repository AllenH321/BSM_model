from __future__ import annotations

from datetime import date, datetime
import pandas as pd

TRADING_DAYS = 252


def to_datetime_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df


def today_date() -> date:
    return datetime.today().date()
