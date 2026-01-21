import pandas as pd
import numpy as np
from prediction_metrics import compute_all_metrics, PredictionMetrics, compute_rolling_ic

def test_perfect_predictions():
    """Metrics should be optimal for perfect predictions."""
    actual = pd.Series([0.05, 0.10, -0.03, 0.08, 0.02, 0.04, -0.01, 0.06, 0.07, 0.09], name='actual')
    predicted = actual.copy()
    lower_ci = predicted - 0.01
    upper_ci = predicted + 0.01
    
    metrics = compute_all_metrics(actual, predicted, lower_ci, upper_ci)
    
    assert metrics.oos_r2 == 1.0
    print(f"DEBUG: IC={metrics.ic}, Type={type(metrics.ic)}")
    assert abs(metrics.ic - 1.0) < 1e-10
    assert metrics.hit_rate == 1.0
    assert metrics.coverage == 1.0
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.bias == 0.0

def test_random_predictions():
    """Metrics should be poor for random predictions."""
    np.random.seed(42)
    # Increase sample size for more stable random metrics
    actual = pd.Series(np.random.randn(100) * 0.10)
    predicted = pd.Series(np.random.randn(100) * 0.10)
    lower_ci = predicted - 0.05
    upper_ci = predicted + 0.05
    
    metrics = compute_all_metrics(actual, predicted, lower_ci, upper_ci)
    
    # Random predictions should have poor R2 (usually <= 0)
    assert metrics.oos_r2 < 0.5 
    assert abs(metrics.ic) < 0.3
    # Hit rate around 50%
    assert 0.4 < metrics.hit_rate < 0.6

def test_insufficient_data():
    """Should return NaN for insufficient data."""
    actual = pd.Series([0.05, 0.10])
    predicted = pd.Series([0.04, 0.11])
    lower_ci = predicted - 0.01
    upper_ci = predicted + 0.01
    
    metrics = compute_all_metrics(actual, predicted, lower_ci, upper_ci)
    
    assert np.isnan(metrics.oos_r2)
    assert metrics.n_observations == 2

def test_bias_calculation():
    """Verify bias calculation (predicted - actual)."""
    actual = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    predicted = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    lower_ci = predicted - 0.05
    upper_ci = predicted + 0.05
    
    metrics = compute_all_metrics(actual, predicted, lower_ci, upper_ci)
    
    # bias = mean(predicted - actual) = 0.05
    assert abs(metrics.bias - 0.05) < 0.0001

def test_hit_rate_sign_only():
    """Hit rate should only care about signs."""
    actual = pd.Series([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    predicted = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    lower_ci = predicted - 0.5
    upper_ci = predicted + 0.5
    
    metrics = compute_all_metrics(actual, predicted, lower_ci, upper_ci)
    assert metrics.hit_rate == 1.0
    
    predicted_opposite = -predicted
    metrics_opposite = compute_all_metrics(actual, predicted_opposite, lower_ci, upper_ci)
    assert metrics_opposite.hit_rate == 0.0

def test_rolling_ic():
    """Verify rolling IC calculation."""
    actual = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    predicted = actual.copy()
    window = 5
    
    roll_ic = compute_rolling_ic(actual, predicted, window=window)
    
    # Perfect correlation should be 1.0
    assert len(roll_ic) == len(actual) - window + 1
    assert all(abs(val - 1.0) < 1e-10 for val in roll_ic)
    assert roll_ic.index[-1] == actual.index[-1]

if __name__ == "__main__":
    test_perfect_predictions()
    test_random_predictions()
    test_insufficient_data()
    test_bias_calculation()
    test_hit_rate_sign_only()
    test_rolling_ic()
    print("All tests passed!")
