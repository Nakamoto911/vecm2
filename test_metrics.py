import pandas as pd
import numpy as np
from backtester import StrategyBacktester

# Mock data
dates = pd.date_range("2020-01-01", periods=13, freq='M')
prices = pd.DataFrame({'EQUITY': [100, 105, 102, 108, 110, 105, 115, 120, 118, 125, 130, 128, 135]}, index=dates)
preds = pd.DataFrame({'EQUITY': [0.05]*13}, index=dates)
ci = pd.DataFrame({'EQUITY': [0.0]*13}, index=dates)
regime = pd.Series([0.1]*13, index=dates)

bt = StrategyBacktester(prices, preds, ci, regime)

# Test with 4% RF
results_4 = bt._calculate_metrics(pd.Series([10000, 10535], index=[dates[0], dates[12]]), pd.DataFrame(), risk_free_rate=0.04)
print(f"Metrics with 4% RF: {results_4}")

# Test with 0% RF
results_0 = bt._calculate_metrics(pd.Series([10000, 10535], index=[dates[0], dates[12]]), pd.DataFrame(), risk_free_rate=0.0)
print(f"Metrics with 0% RF: {results_0}")

# Mock returns series for internal logic check
equity_curve = pd.Series([100, 105, 102, 108], index=dates[:4])
# returns: 0.05, -0.02857, 0.0588
metrics = bt._calculate_metrics(equity_curve, pd.DataFrame(), risk_free_rate=0.04)
print(f"Detailed Metrics: {metrics}")
