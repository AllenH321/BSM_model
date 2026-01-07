import yfinance as yf

def validate_ticker(ticker: str) -> bool:
    """
    Validate a stock ticker by checking if yfinance can return price data.
    """
    try:
        info = yf.Ticker(ticker).info
        return "regularMarketPrice" in info
    except Exception:
        return False
