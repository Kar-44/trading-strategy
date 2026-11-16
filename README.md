# trading-strategy

Simple Pairs-trading Strategy using Machine Learning Scikit-learn package to enhance results. 

## How it works?

RandomForestClassifier is first trained on: absolute spread Z-scores, 3-day Z-score velocity, short-to-long volatility ratios, rolling beta instability and spread autocorrelation. 

The strategy strictly prevents look-ahead bias using 70/30 chronological train/test split with 5-day purged gap. It also stops trade
execution when pairs split and resumes when they merge again. The ML implementation boosts the average Sharpe ratio
by 52% (from 0.48 to 0.73, peak at 1.14) and increased the average Annualized return from 5.1% to 11.8%.

The 'base strategy' is also in this script as to show how much work the Machine Learning does. The results are presented using a summary table and graphs.

## How to run the script?

Make sure you have downloaded the following packages: 

```pip install numpy pandas scikit-learn yfinance matplotlib```

Change the tickers to comapnies' of your interest in "__main__" at the bottom of the script, you can tweak the parameters throughout the code.

Run using command:
```python3 TradStrat.py```

Results sample for Google and Meta (graph in files):
```
Downloading data for GOOGL and META...

---ML feature importance (on 70% data)---
feat_abs_z               : 35.63%
feat_z_velocity          : 13.22%
feat_vol_regime          : 20.00%
feat_beta_instability    : 16.84%
feat_autocorr            : 14.31%
---Test set: 2024-04-26 to 2026-10-30---

--- Baseline StatArb (No ML) ---
Total Test Return : 10.52%
Annualized Return : 4.14%
Annualized Vol    : 5.47%
Sharpe Ratio      : 0.76
Max Drawdown      : -4.98%

--- ML-Enhanced StatArb ---
Total Test Return : 18.46%
Annualized Return : 7.52%
Annualized Vol    : 12.38%
Sharpe Ratio      : 0.61
Max Drawdown      : -11.42%
```

