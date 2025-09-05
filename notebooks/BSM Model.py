import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model

# Import data
ticker = 'DIS'
start_date = '2020-01-01'
end_date = '2025-05-03'  # Use latest available before forecast
hist_data = yf.download(ticker, start=start_date, end=end_date)
hist_data['log_ret'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
returns = hist_data['log_ret'].dropna() * 100

# GARCH(1,1)
am = arch_model(returns, vol='Garch', p=1, q=1)  # specifies the GARCH(1,1) model
res = am.fit(disp='off')  # without displaying intermediate output

# Calculate horizon - business days to 2025-05-09 and 2025-05-23
future_dates = pd.bdate_range(start=returns.index[-1], end='2025-05-23')
target_indices = {
    '2025-05-09': np.where(future_dates == '2025-05-09')[0][0],
    '2025-05-23': np.where(future_dates == '2025-05-23')[0][0],
}
horizon = target_indices['2025-05-23'] + 1

# Forecast volatility
forecast = res.forecast(horizon=horizon)  # generates the volatility forecasts for the specified horizon
# extracts the variance forecasts, calculate the standard deviation, and scales back to the percentage format
vol_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100

# Print volatility
print("\nGARCH(1,1) Forecasted Volatility:")
for date_str, idx in target_indices.items():
    print(f"{date_str}: {vol_forecast[idx]:.4f}")


from scipy.stats import norm

# BSM Model
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


# BSM European Call Strikes:80, 93, 100, Maturities: 9/5/25 and 23/5/25
S = hist_data['Close'].iloc[-1]
strikes = [80, 93, 100]
r = 0.007
maturities = {
    '2025-05-09': 0,
    '2025-05-23': 1
}

print('Black-Scholes Call Prices:\n')

for date_idx, vol_idx in maturities.items():
    # Compute time to maturity in years (using business days)
    T = len(pd.bdate_range('2025-05-03', date_idx)) / 252
    sigma = vol_forecast[vol_idx]

    print(f"--- Maturity: {date_idx} ---")
    for K in strikes:
        call_price = bs_call(S, K, T, r, sigma)
        print(f"Strike ${K}: {float(call_price.iloc[0]):.4f}")
    print()


# BSM European Put Strikes:80, 93, 100, Maturities: 9/5/25 and 23/5/25
S = hist_data['Close'].iloc[-1]
strikes = [80, 93, 100]
r = 0.007
maturities = {
    '2025-05-09': 0,
    '2025-05-23': 1
}

print('Black-Scholes Put Prices:\n')

for date_idx, vol_idx in maturities.items():
    # Compute time to maturity in years (using business days)
    T = len(pd.bdate_range('2025-05-03', date_idx)) / 252
    sigma = vol_forecast[vol_idx]

    print(f"--- Maturity: {date_idx} ---")
    for K in strikes:
        put_price = bs_put(S, K, T, r, sigma)
        print(f"Strike ${K}: {float(put_price.iloc[0]):.4f}")
    print()