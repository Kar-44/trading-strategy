import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# STEP 1: DATA LOADER (Synthetic Cointegrated Pair + Real Data Option)
# =====================================================================
def get_market_data(n_days=1500, seed=42):
    """
    Generates two realistic, cointegrated price series (Asset X and Asset Y)
    with a slowly drifting hedge ratio so we can test our dynamic regression.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")

    # Simulate Asset X as a geometric random walk (like a benchmark ETF)
    returns_x = np.random.normal(loc=0.0003, scale=0.012, size=n_days)
    price_x = 100 * np.exp(np.cumsum(returns_x))

    # Create a slowly drifting true hedge ratio (beta) between 1.1 and 1.4
    true_beta = 1.25 + 0.15 * np.sin(np.linspace(0, 3 * np.pi, n_days))

    # Simulate a mean-reverting spread (Ornstein-Uhlenbeck process)
    spread_noise = np.zeros(n_days)
    theta = 0.08  # Speed of mean reversion
    for t in range(1, n_days):
        spread_noise[t] = (1 - theta) * spread_noise[t - 1] + np.random.normal(0, 0.8)

    # Asset Y is driven by Asset X, the drifting beta, and the mean-reverting spread
    price_y = 15.0 + true_beta * price_x + spread_noise

    df = pd.DataFrame({"Asset_X": price_x, "Asset_Y": price_y}, index=dates)

    # -----------------------------------------------------------------
    # OPTIONAL: To use real data, un/comment the lines below:
    import yfinance as yf
    raw = yf.download(["KO", "PEP"], start="2019-01-01", end="2026-01-01")["Close"]
    df = raw.rename(columns={"KO": "Asset_X", "PEP": "Asset_Y"}).dropna()
    # -----------------------------------------------------------------

    return df


# =====================================================================
# STEP 2: CUSTOM ROLLING RIDGE REGRESSION (The "Custom Regression Curve")
# =====================================================================
def rolling_weighted_ridge_regression(x_series, y_series, window=60, l2_penalty=1.0, half_life=20):
    """
    Fits a custom rolling linear regression: Y = alpha + beta * X
    Uses Exponentially Weighted Least Squares + L2 (Ridge) Regularization.

    1. Strictly uses [t - window : t] data (prevents Look-ahead Bias).
    2. Exponential weights prioritize recent market regime over old data.
    3. L2 penalty prevents beta from exploding during market crashes.
    """
    n = len(x_series)
    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)

    x_vals = x_series.values
    y_vals = y_series.values

    # Pre-calculate exponential decay weights for the rolling window
    # Older observations get smaller weights; the newest observation gets weight = 1.0
    decay_rate = np.log(2) / half_life
    weights = np.exp(-decay_rate * np.arange(window - 1, -1, -1))
    W = np.diag(weights)  # Diagonal weight matrix

    # Regularization matrix (we penalize the slope 'beta', not the intercept 'alpha')
    reg_matrix = np.array([[0.0, 0.0],
                           [0.0, l2_penalty]])

    # Loop strictly forward in time starting once we have enough history
    for t in range(window - 1, n):
        # Slice ONLY historical data up to and including day t
        x_win = x_vals[t - window + 1: t + 1]
        y_win = y_vals[t - window + 1: t + 1]

        # Design matrix X with a column of 1s for the intercept (alpha)
        X_mat = np.column_stack([np.ones(window), x_win])

        # Closed-form Weighted Ridge Regression solution:
        # theta = (X^T * W * X + lambda * I)^(-1) * (X^T * W * Y)
        Xt_W = X_mat.T @ W
        lhs = Xt_W @ X_mat + reg_matrix
        rhs = Xt_W @ y_win

        # Solve linear system (faster and more numerically stable than matrix inversion)
        alpha_t, beta_t = np.linalg.solve(lhs, rhs)

        alphas[t] = alpha_t
        betas[t] = beta_t

    return pd.Series(alphas, index=x_series.index), pd.Series(betas, index=x_series.index)


# =====================================================================
# STEP 3: SIGNAL GENERATION & VOLATILITY TARGETING
# =====================================================================
def run_stat_arb_strategy(
        df,
        reg_window=60,
        z_window=30,
        l2_penalty=2.0,
        half_life=20,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        target_vol=0.10,
        tc_bps=5.0
):
    """
    Executes the full backtest pipeline with hysteresis, vol-scaling, and fees.
    """
    data = df.copy()

    # 1. Estimate dynamic alpha and beta using ONLY past data at each step
    data["alpha"], data["beta"] = rolling_weighted_ridge_regression(
        data["Asset_X"], data["Asset_Y"],
        window=reg_window, l2_penalty=l2_penalty, half_life=half_life
    )

    # 2. Compute the live equilibrium spread (residuals of the custom regression)
    data["spread"] = data["Asset_Y"] - (data["alpha"] + data["beta"] * data["Asset_X"])

    # 3. Compute rolling Z-score of the spread using strictly past window
    spread_mean = data["spread"].rolling(window=z_window).mean()
    spread_std = data["spread"].rolling(window=z_window).std()
    data["z_score"] = (data["spread"] - spread_mean) / spread_std

    # 4. Generate stateful trading signals with Hysteresis (Buffer Zones) and Stop-Loss
    # +1 = Long Spread (Buy Y, Sell beta*X)
    # -1 = Short Spread (Sell Y, Buy beta*X)
    #  0 = Flat (In Cash)
    n = len(data)
    z_vals = data["z_score"].values
    raw_signal = np.zeros(n)
    current_pos = 0

    for t in range(n):
        z = z_vals[t]
        if np.isnan(z):
            raw_signal[t] = 0
            continue

        if current_pos == 0:
            if z < -entry_z and z > -stop_z:
                current_pos = 1  # Spread is abnormally cheap -> Buy Spread
            elif z > entry_z and z < stop_z:
                current_pos = -1  # Spread is abnormally expensive -> Short Spread
        elif current_pos == 1:
            # Exit if spread reverts to normal (-exit_z) OR blows through stop-loss (-stop_z)
            if z >= -exit_z or z <= -stop_z:
                current_pos = 0
        elif current_pos == -1:
            # Exit if spread reverts to normal (+exit_z) OR blows through stop-loss (+stop_z)
            if z <= exit_z or z >= stop_z:
                current_pos = 0

        raw_signal[t] = current_pos

    data["signal"] = raw_signal

    # 5. Volatility Targeting (Inverse Volatility Position Sizing)
    # Calculate daily return of 1 unit of the hedged spread pair
    ret_x = data["Asset_X"].pct_change()
    ret_y = data["Asset_Y"].pct_change()

    # Capital allocation weights per unit of spread: 1 part Y, beta parts X
    # We normalize by (1 + |beta|) so gross leverage before vol-scaling equals 1.0
    gross_exposure = 1.0 + data["beta"].abs()
    weight_y = 1.0 / gross_exposure
    weight_x = -data["beta"] / gross_exposure

    # Note: We lag weight_x and weight_y by 1 day when computing historical spread volatility
    raw_pair_return = weight_y.shift(1) * ret_y + weight_x.shift(1) * ret_x

    # Annualized rolling volatility of the spread (252 trading days)
    rolling_ann_vol = raw_pair_return.rolling(window=z_window).std() * np.sqrt(252)

    # Scale position size to hit 'target_vol' (cap leverage at 2.0x for safety)
    vol_scalar = (target_vol / rolling_ann_vol).clip(upper=2.0).fillna(0.0)
    data["sized_position"] = data["signal"] * vol_scalar

    # =================================================================
    # CRITICAL NO-LOOKAHEAD STEP: SHIFT POSITIONS BY 1 DAY (.shift(1))
    # =================================================================
    # A signal computed using Day T's closing prices can ONLY be executed
    # at Day T's close / Day T+1's open, earning Day T+1's return.
    data["exec_weight_y"] = (data["sized_position"] * weight_y).shift(1).fillna(0.0)
    data["exec_weight_x"] = (data["sized_position"] * weight_x).shift(1).fillna(0.0)

    # 6. Calculate Gross Returns, Transaction Costs, and Net Returns
    data["gross_ret"] = data["exec_weight_y"] * ret_y + data["exec_weight_x"] * ret_x

    # Turnover is the total absolute change in portfolio weights on both legs
    turnover = data["exec_weight_y"].diff().abs().fillna(0.0) + data["exec_weight_x"].diff().abs().fillna(0.0)
    data["tc_cost"] = turnover * (tc_bps / 10000.0)

    data["net_ret"] = data["gross_ret"] - data["tc_cost"]
    data["cum_gross"] = (1 + data["gross_ret"]).cumprod()
    data["cum_net"] = (1 + data["net_ret"]).cumprod()

    return data


# =====================================================================
# STEP 4: INSTITUTIONAL PERFORMANCE METRICS & VISUALIZATION
# =====================================================================
def evaluate_performance(data):
    """
    Computes key quant interview metrics: Sharpe, Sortino, Max Drawdown, Calmar.
    """
    valid = data.dropna(subset=["z_score"]).copy()
    net_r = valid["net_ret"]

    ann_ret = net_r.mean() * 252
    ann_vol = net_r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (only penalizes downside volatility)
    downside_r = net_r[net_r < 0]
    downside_vol = downside_r.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0.0

    # Maximum Drawdown
    equity = (1 + net_r).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    print("=== BACKTEST PERFORMANCE SUMMARY (NET OF FEES) ===")
    print(f"Annualized Return : {ann_ret:.2%}")
    print(f"Annualized Vol    : {ann_vol:.2%}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Sortino Ratio     : {sortino:.2f}")
    print(f"Maximum Drawdown  : {max_dd:.2%}")
    print(f"Calmar Ratio      : {calmar:.2f}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(valid.index, valid["cum_gross"], label="Gross Cumulative Return", linestyle="--", alpha=0.7)
    axes[0].plot(valid.index, valid["cum_net"], label="Net Cumulative Return (After Fees)", linewidth=2)
    axes[0].set_title("Strategy Equity Curve (Gross vs. Net)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid.index, valid["z_score"], label="Spread Z-Score", color="purple", alpha=0.8)
    axes[1].axhline(2.0, color="red", linestyle="--", label="Entry Threshold (+/- 2.0)")
    axes[1].axhline(-2.0, color="red", linestyle="--")
    axes[1].axhline(0.5, color="green", linestyle=":", label="Exit Threshold (+/- 0.5)")
    axes[1].axhline(-0.5, color="green", linestyle=":")
    axes[1].set_title("Rolling Spread Z-Score & Hysteresis Bands")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(valid.index, valid["beta"], label="Dynamic Ridge Hedge Ratio (Beta)", color="teal")
    axes[2].set_title("Custom Rolling Weighted Ridge Regression Beta")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    market_df = get_market_data(n_days=1500)
    results_df = run_stat_arb_strategy(market_df)
    evaluate_performance(results_df)