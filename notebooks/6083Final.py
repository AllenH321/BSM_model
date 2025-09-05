import numpy as np
from scipy.stats import norm


S = 110       # Spot price
K = 110       # Strike price
T = 0.25      # Time to maturity (years)
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility


# Part 1. Implement the binomial option pricing model to price European and American options on DIS stock


def binomial_option(S, K, T, r, sigma, n, option_type="call", american=False):
    dt = T/n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)

    prices = np.zeros(n+1)
    for i in range(n+1):
        prices[i] = S * (u**(n-i)) * (d**i)

    option = np.zeros(n+1)
    for i in range(n+1):
        if option_type == "call":
            option[i] = max(0, prices[i] - K)
        else:
            option[i] = max(0, K - prices[i])

    for j in range(n-1, -1, -1):
        for i in range(j+1):
            option[i] = np.exp(-r*dt)*(p*option[i] + (1-p)*option[i+1])
            if american:
                prices[i] = prices[i]*d
                if option_type == "call":
                    option[i] = max(option[i], prices[i] - K)
                else:
                    option[i] = max(option[i], K - prices[i])
    return option[0]

binomial_price = binomial_option(S, K, T, r, sigma, n=100, option_type="call", american=False)
print(f"Binomial Call Price (100000 steps): {binomial_price:.2f}")


# Part 2. Compare the results with the Black-Scholes model and asses the convergence properties of the Binomial model

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

bs_price = bs_call(S, K, T, r, sigma)
print(f"Black-Scholes Call Price: {bs_price:.2f}")


import time

# --- Optimized Binomial Option Pricing Function ---
def fast_binomial_option(S, K, T, r, sigma, n, option_type="call", american=False):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal asset prices
    j = np.arange(n + 1)
    ST = S * (u ** (n - j)) * (d ** j)

    # Terminal option values
    if option_type == "call":
        option = np.maximum(0, ST - K)
    else:
        option = np.maximum(0, K - ST)

    # Backward induction
    for i in range(n, 0, -1):
        option = disc * (p * option[:-1] + (1 - p) * option[1:])
        if american:
            ST = ST[:-1] / u
            if option_type == "call":
                option = np.maximum(option, ST - K)
            else:
                option = np.maximum(option, K - ST)

    return option[0]

# --- Parameters ---
S = 110         # Current stock price
K = 110         # Strike price
T = 30 / 365    # Time to maturity in years
r = 0.05        # Risk-free rate
sigma = 0.2     # Volatility

# --- Precompute over range of steps ---
precomputed_steps = list(range(50, 5001, 50))  # 100 steps total

start = time.time()

precomputed_prices = [
    fast_binomial_option(S, K, T, r, sigma, n, option_type="call", american=False)
    for n in precomputed_steps
]

end = time.time()
print(f"Precomputation completed in {end - start:.2f} seconds.")

from ipywidgets import interact, IntSlider

# Black-Scholes reference value for comparison

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

bs_price = bs_call(S, K, T, r, sigma)

"""
# Define the interactive plot
@interact(n=IntSlider(min=precomputed_steps[0], max=precomputed_steps[-1], step=50, value=500, description='Steps (n)'))
def plot_convergence(n):
    idx = precomputed_steps.index(n) + 1
    steps = precomputed_steps[:idx]
    prices = precomputed_prices[:idx]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, prices, label='Binomial Price (fast)')
    plt.axhline(bs_price, color='red', linestyle='--', label='Black-Scholes Price')
    plt.xlabel('Number of Steps')
    plt.ylabel('Option Price')
    plt.title('Convergence of Binomial Model to Black-Scholes Price')
    plt.legend()
    plt.grid(True)
"""

# Compute binomial prices for fewer steps
import matplotlib.pyplot as plt
precomputed_steps = list(range(50, 1501, 10))  # Fewer steps for clean graph
precomputed_prices = [
    fast_binomial_option(S, K, T, r, sigma, n, option_type="call", american=False)
    for n in precomputed_steps
]

# Plot the convergence
plt.figure(figsize=(10, 5))
plt.plot(precomputed_steps, precomputed_prices, label='Binomial Price (fast)', marker='o')
plt.axhline(bs_price, color='red', linestyle='--', label=f'Black-Scholes Price (${bs_price:.2f})')
plt.xlabel('Number of Steps')
plt.ylabel('Option Price')
plt.title('Convergence of Binomial Model to Black-Scholes Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure as a PNG
plt.savefig("binomial_vs_bs_convergence.png", dpi=300)
plt.show()


# Part 3. Get the historical stock price data for DIS and calculate the historical volatility

import matplotlib.pyplot as plt
import yfinance as yf

# Load data for DIS stock if not already loaded
hist_data = yf.download("DIS", start="2020-01-01")

# Calculate daily log returns
hist_data['log_ret'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))

# Calculate 21-day rolling (monthly) volatility, annualized
hist_data['volatility'] = hist_data['log_ret'].rolling(window=21).std() * np.sqrt(252)

# Dual-axis plot
fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot Close price
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price', color=color)
ax1.plot(hist_data.index, hist_data['Close'], color=color, label='Close Price')
ax1.tick_params(axis='y', labelcolor=color)

# Plot volatility on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('21-Day Rolling Volatility (Annualized)', color=color)
ax2.plot(hist_data.index, hist_data['volatility'], color=color, linestyle='--', label='21-Day Volatility')
ax2.tick_params(axis='y', labelcolor=color)

# Title and formatting
fig.suptitle('DIS Price vs. 21-Day Rolling Volatility')
fig.tight_layout()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(hist_data['volatility'], label='21-Day Rolling Volatility')
plt.title('Historical Volatility of DIS')
plt.legend()
plt.grid()
plt.show()


# Part 4. Implement a GARCH(1,1) model to forecast future volatility
from arch import arch_model
import pandas as pd


returns = hist_data['log_ret'].dropna() * 100
am = arch_model(returns, vol='GARCH', p=1, q=1)
res = am.fit(disp='off')

# Calculate horizon - business days to 2025-05-09 and 2025-05-23
future_dates = pd.bdate_range(start=returns.index[-1], end='2025-05-23')
target_indices = {
    '2025-05-09': np.where(future_dates == '2025-05-09')[0][0],
    '2025-05-23': np.where(future_dates == '2025-05-23')[0][0],
}
horizon = target_indices['2025-05-23'] + 1

# Forecast volatility
forecast = res.forecast(horizon=horizon)
vol_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100

# Print volatility
print("\nGARCH(1,1) Forecasted Volatility:")
for date_str, idx in target_indices.items():
    print(f"{date_str}: {vol_forecast[idx]:.4f}")


# Part 5. Use the forecasted volatility in the Black-Scholes model to price options and compare with the market prices
# Import data
ticker = 'DIS'
start_date = '2020-01-01'
end_date = '2025-05-03'  # Use latest available before forecast
hist_data = yf.download(ticker, start=start_date, end=end_date)
hist_data['log_ret'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
returns = hist_data['log_ret'].dropna() * 100

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


