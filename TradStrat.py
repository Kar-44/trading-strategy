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

    # Purging boundary makes is so that train set does not leak into the test set.
    train_df = data.ilox[: split_idx - lookforward_horizon]

    X_train = train_df[feature_cols].values
    y_train = train_df["ml_target"].values

    # Implementing shallow trees (max_depth=4) and high min_sample_leaf (25) prevents overfitting to noise
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=25,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Predict win probability across the timeline 70/30 in and out of sample
    X_all = data[feature_cols].values
    data["ml_win_prob"] = rf_model.predict_proba(X_all)[:, 1]

    #Print feature importances
    print("\n---ML feature importance (on 70% data)---")
    for col, imp in zip(feature_cols, rf_model.feature_importances_):
        print(f"{col:<25}: {imp:.2%}")

    # Trading loop (Baseline vs ML-enhanced)
    z_vals = data["z_score"].values
    prob_vals = data["ml_win_prob"].values

    ml_signal = np.zeros(n_total)
    base_signal = np.zeros(n_total)

    curr_ml = 0.0
    curr_base = 0.0

    for t in range(n_total):
        z = z_vals[t]
        prob = prob_vals[t]

        # Baseline strategy (fixed z= 2.0, no ML)
        if curr_base == 0:
            if -stop_z < z < -2.0:
                curr_base = 1.0
            elif 2.0 < z < stop_z:
                curr_base = -1.0
        elif curr_base == 1.0 and (z >= -0.5 or z <= -stop_z):
            curr_base = 0.0
        elif curr_base == -1.0 and (z <= -0.5 or z >= stop_z):
            curr_base = 0.0
        base_signal[t] = curr_base

        # ML-enhanced strategy (lower entry Z, ML filter, Conviction sizing)
        if curr_ml == 0:
            if -stop_z < z < -entry_z and prob >= ml_prob_threshold:
                # Scale conviction between 0.8 and 1.6 based on ML confidence
                conviction = np.clip(1.0 + (prob - ml_prob_threshold) * 4.0, 0.8, 1.6)
                curr_ml = 1.0 * conviction
            elif entry_z < z < stop_z and prob >= ml_prob_threshold:
                conviction = np.clip(1.0 + (prob - ml_prob_threshold) * 4.0, 0.8, 1.6)
                curr_ml = -1.0 * conviction
        elif curr_ml > 0 and (z >= -exit_z or z <= -stop_z):
            curr_ml == 0.0
        elif curr_ml < 0 and (z <= -exit_z or z >= stop_z):
            curr_ml = 0.0
        ml_signal[t] = curr_ml

    data["base_signal"] = base_signal
    data['ml_signal'] = ml_signal

    # Portfolio weights, volatility scaling and strict 1d lag
    ret_x = data["Price_X"].pct_change().fillna(0.0)
    ret_y = data["Price_Y"].pct_change().fillna(0.0)

    #Dollar-neutral leg weights per unit spread
    gross_unit = 1.0 + data["beta"].abs()
    w_y = 1.0 / gross_unit
    w_x = -data["beta"] / gross_unit

    # Calculate rolling annual. volatility of 1 unit spread
    unit_spread_ret = w_y.shift(1) * ret_y + w_x.shift(1) * ret_x
    ann_spread_vol = unit_spread_ret.rolling(25).std() * np.sqrt(252)

    # Inverse volatility scalar capped at max_leverage
    vol_scalar = (target_vol / ann_spread_vol).clip(upper=max_leverage).fillna(0.0)

    # Compute net returns for both- Baseline and ML-enhanced
    for prefix, sig_col in [("base", "base_signal"), ("ml", "ml_signal")]:
        exec_wy = (data[sig_col] * vol_scalar * w_y).shift(1).fillna(0.0)
        exec_wx = (data[sig_col] * vol_scalar * w_x).shift(1).fillna(0.0)

        gross_r = exec_wy * ret_y + exec_wx * ret_x
        turnover = exec_wy.diff().abs().fillna(0.0) + exec_wx.diff().abs().fillna(0.0)
        tc = turnover * (tc_bps / 10000.0)

        data[f"{prefix}_net_ret"] = gross_r - tc

    return data, split_idx

# 5. Evaluate test set (30%) out of sample and plots
def eval_and_plot(data, split_idx):
    """
    Evaluates strict Out-of-sample metrics and plots curves.
    :param data:
    :param split_idx:
    :return:
    """
    test_df = data.iloc[split_idx:].copy()
    split_date = test_df.index[0]

    print(f"---Test set: {split_date.date()} to {test_df.index[-1].date()}---")

    for label, col in [("Baseline StatArb (No ML)", "base_net_ret"), ("ML-Enhanced StatArb", "ml_net_ret")]:
        r = test_df[col]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        eq = (1 + r).cumprod()
        max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
        total_ret = eq.iloc[-1] - 1.0

        print(f"\n--- {label} ---")
        print(f"Total Test Return : {total_ret:.2%}")
        print(f"Annualized Return : {ann_ret:.2%}")
        print(f"Annualized Vol    : {ann_vol:.2%}")
        print(f"Sharpe Ratio      : {sharpe:.2f}")
        print(f"Max Drawdown      : {max_dd:.2%}")

        # Plot full timeline + Out-of-Sample test performance
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top chart: Out-of-Sample 30% Test set direct comparison
    test_base_eq = (1 + test_df["base_net_ret"]).cumprod()
    test_ml_eq = (1 + test_df["ml_net_ret"]).cumprod()

    axes[0].plot(test_df.index, test_base_eq, label="Baseline StatArb (30% Test Set)", linestyle="--", color="gray")
    axes[0].plot(test_df.index, test_ml_eq, label="ML-Enhanced StatArb (30% Test Set)", linewidth=2.2,
                   color="royalblue")
    axes[0].set_title("Out-of-Sample Test Set Performance (Latest 30% Unseen Data - Net of Fees)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom chart: Full timeline showing the 70/30 Train/Test split
    full_ml_eq = (1 + data["ml_net_ret"]).cumprod()
    axes[1].plot(data.index, full_ml_eq, label="ML-Enhanced Equity Curve (Full Timeline)", color="teal")
    axes[1].axvline(split_date, color="crimson", linestyle="--", linewidth=2, label="70% Train / 30% Test Split")
    axes[1].set_title("Full Strategy Timeline with 70% Train / 30% Test Boundary")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw_df = fetch_pair_data(ticker_x="XOM", ticker_y="CVX")
    feat_df, feat_cols = eng_ml_features(raw_df)
    bt_df, split_point = train_ml_strategy(feat_df, feat_cols)
    eval_and_plot(bt_df, split_point)

# End of script