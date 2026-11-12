import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1. Real market data loading from yfinance
def fetch_pair_data(ticker_x="XOM", ticker_y="CVX", start="2018-01-01", end="2026-11-01"):
    """
    :param ticker_x:
    :param ticker_y:
    :param start:
    :param end:
    Downloads historical financial data from Yahoo Finance, by Default it is data of a
    pair ExxonMobil and Chevron, economically linked energy giants. Between 2018-01-01 and 2026-11-01
    """
    print(f"Downloading data for {ticker_x} and {ticker_y}...")
    raw = yf.download([ticker_x, ticker_y], start=start, end=end, auto_adjust=True, progress=False)

    # Extract Close prices from different yfinance versions, removing missing values to prevent errors
    close_prices = raw["Close"] if "Close" in raw else raw df = pd.DataFrame({
        "Price_X": close_prices[ticker_x],
        "Price_Y": close_prices[ticker_y]
    }).dropna()

    # Convert to Log-Prices so beta represents percentage elasticity
    df["Log_X"] = np.log(df["Price_X"])
    df["Log_Y"] = np.log(df["Price_Y"])

    return df

# 2. Custom rolling Ridge regression on Log-Prices
def rolling_weighted_ridge_reg(x_series, y_series, window=60, l2_penalty=0.5, half_life=20):
    """
    :param x_series:
    :param y_series:
    :param window:
    :param l2_penalty:
    :param half_life:
    :return:
    Fits Log_Y = alpha + beta * Log_X using only data within the rolling window.
    Includes exponential time-decay and L2 (Ridge) regularization
    """
    # Set
    n=len(x_series)
    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)

    x_vals = x_series.values
    y_vals = y_series.values

    # Exponential weights, ie. recent days matter more than early ones
    # ==============
    # Potentially to change for more optimized option that does not create a giant matrix
    decay_rate = np.log(2) / half_life
    weights = np.exp(-decay_rate * np.arange(window -1, -1, -1)) # np.arange(start, stop, step)
    W = np.diag(weights) # turn 1D array into a Matrix with diagonal weights and 0s elsewhere

    # Penalize slope to prevent giant hedge ratios
    reg_matrix = np.array([[0.0, 0.0],
                            [0.0, l2_penalty]])

    for t in range(window-1, n):
        x_win = x_vals[t - window+1 : t+1]
        y_win = y_vals[t - window+1 : t+1]

        X_mat = np.column_stack([np.ones(window), x_win])
        Xt_W = X_mat.T @ W

        # Solve (X^T * W * X + lambda * I) * theta = X^T * W * Y
        alpha_t, beta_t = np.linalg.solve(Xt_W @ X_mat + reg_matrix, Xt_W @ y_win)
        alphas[t] = alpha_t
        betas[t] = beta_t
    return pd.Series(alphas, index=x_series.index), pd.Series (betas, index=x_series.index)

# 3. Machine Learning features
def eng_ml_features(df, reg_window=60, z_window=25, lookforward_horizon=5):
    """
    :param df:
    :param reg_window:
    :param z_window:
    :param lookforward_horizon:
    :return:
    Creates stationary, causal features at Day T and labels whether a mean-reversion
    trade would have been profitable over the next N days.
    """
    data = df.copy()
    # a. Compute dynamic log-spread from custom regression curve
    data["alpha"], data["beta"] = rolling_weighted_ridge_reg(data["Log_X"], data["Log_Y"], window=reg_window)

    data["spread"] = data["Log_Y"] - (data["alpha"]+data["beta"] * data["Log_X"])

    # b. Rolling z-score of the spread
    spread_mean = data["spread"].rolling(z_window).mean()
    spread_std = data["spread"].rolling(z_window).std()
    data["z_score"] = (data["spread"] - spread_mean) / spread_std

    # 5 ML features using only past/present data at time t
    # 3.1. Absolute Z-Score (magnitude of mispricing)
    data["feat_abs_z"] = data["z_score"].abs()

    # 3.2. 3-day Z-Score velocity (is the spread accelerating or turning back)
    data["feat_z_velocity"] = data["z_score"].diff(3) * np.sign(data["z_score"])

    # 3.3. Volatility regime ratio (short-term 10d vol/ long-term 60d vol)
    vol_10 = data["spread"].diff().rolling(10).std()
    vol_60 = data["spread"].diff().rolling(60).std()
    data["feat_vol_regime"] = vol_10 / vol_60

    # 3.4. Hedge ratio (beta) stability (10d roll std of beta)
    # High instability means pair relationship is breaking down
    data["feat_beta_instability"] = data["beta"].rolling(10).std()

    # 3.5. Spread autocorrelation (15d roll lag-1 correlation of spread changes)
    # Negative autocorrelation means strong mean-reverting regime; Positive- trending regime
    spread_diff = data["spread"].diff()
    data["feat_autocorr"] = spread_diff.rolling(15).corr(spread_diff.shift(1))

    # Target Labeling (used for training on 70% of dataset)
    # If z_score < 0, we want spread to rise (+1). If z_score > 0 we want spread to fall (-1)
    intended_dir = -np.sign["z_score"]
    future_spread_change = data["spread"].shift(-lookforward_horizon) - data["spread"]
    # Label = 1 if spread moves in mean-reverting direction over the next 'lookforward_horizon' days
    data["ml_target"] = ((intended_dir * future_spread_change) > 0).astype(int)

    feature_cols = [
        "feat_abs_z",
        "feat_z_velocity",
        "feat_vol_regime",
        "feat_beta_instability",
        "feat_autocorr"
    ]

    return data.dropna(subset=feature_cols + ["z_score"]).copy(), feature_cols

# 4. 70/30 purged train/test split and ML model
def train_ml_strategy(
data,
    feature_cols,
    train_ratio=0.70,
    lookforward_horizon=5,
    entry_z=1.15,          # Lower entry threshold to eliminate cash drag
    exit_z=0.25,           # Hold winners closer to full mean reversion
    stop_z=3.75,           # Stop-loss for structural breaks
    ml_prob_threshold=0.52,# Minimum ML win probability required to take a trade
    target_vol=0.20,       # 20% annualized target volatility
    max_leverage=2.5,      # Max gross leverage (standard Reg-T margin)
    tc_bps=4.0             # 4 bps transaction cost per trade
):
    """
    Trains a Random Forest Meta-Model on the first 70% of data (with purge gap)
    and evaluates performance on the 30% latest test set.
    """
    n_total = len(data)
    split_idx = int(n_total * train_ratio)
    