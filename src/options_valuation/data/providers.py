import datetime as dt
import time
import requests
import yfinance as yf
import pandas as pd

from options_valuation.config import get_key

def fetch_alpha_vantage(
    ticker: str,
    api_key: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch daily adjusted time series from Alpha Vantage.
    Returns a DataFrame with columns: date, open, high, low, close, adj_close, volume
    Raises RuntimeError on API errors / empty data.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    print(payload)

    # Common Alpha Vantage failure modes
    if "Error Message" in payload:
        raise RuntimeError(payload["Error Message"])
    if "Note" in payload:
        # Often rate limit
        raise RuntimeError(payload["Note"])
    if "Time Series (Daily)" not in payload:
        raise RuntimeError(f"Unexpected response keys: {list(payload.keys())}")

    ts = payload["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert & rename columns
    df = df.rename(columns = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
    })[["open", "high", "low", "close", "adj_close", "volume"]]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Data filter
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        raise RuntimeError("Alpha Vantage returned no rows after date filtering.")\

    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df

def fetch_yfinance(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=str(start),
        end=str(end + dt.timedelta(days=1)),
        auto_adjust=False,
        progress=False,
        group_by="column",   # helps consistency
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data.")

    # If MultiIndex columns, flatten them (e.g., ('Open','DIS') -> 'open')
    if isinstance(df.columns, pd.MultiIndex):
        # Usually level 0 is 'Open/High/Low/Close/Adj Close/Volume'
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Now safe: columns are strings
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Ensure expected columns
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance missing columns: {missing}. Got: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[expected]

def get_alpha_vantage_key(env_var: str = "API_KEY") -> str:
    key = os.getenv(env_var, "").strip()
    if not key:
        raise EnvironmentError(
            f"Missing AlphaVantage key. Set environment variable {env_var} in your .env or shell."
        )
    return key