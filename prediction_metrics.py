"""
prediction_metrics.py
Prediction quality metrics for macro return forecasting.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class PredictionMetrics:
    """Container for all prediction quality metrics."""
    # Primary
    oos_r2: float
    ic: float
    ic_pvalue: float
    hit_rate: float
    coverage: float
    
    # Secondary
    rmse: float
    mae: float
    bias: float
    interval_width: float
    ic_tstat: float
    
    # Metadata
    n_observations: int
    nominal_coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'oos_r2': self.oos_r2,
            'ic': self.ic,
            'ic_pvalue': self.ic_pvalue,
            'hit_rate': self.hit_rate,
            'coverage': self.coverage,
            'rmse': self.rmse,
            'mae': self.mae,
            'bias': self.bias,
            'interval_width': self.interval_width,
            'ic_tstat': self.ic_tstat,
            'n_observations': self.n_observations,
            'nominal_coverage': self.nominal_coverage
        }


def compute_all_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    lower_ci: pd.Series,
    upper_ci: pd.Series,
    nominal_coverage: float = 0.90
) -> PredictionMetrics:
    """
    Compute all prediction quality metrics.
    
    Args:
        actual: Realized returns
        predicted: Point predictions
        lower_ci: Lower confidence interval bound
        upper_ci: Upper confidence interval bound
        nominal_coverage: Expected coverage level (e.g., 0.90 for 90% CI)
    
    Returns:
        PredictionMetrics dataclass with all computed metrics
    """
    # Align and clean
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }).dropna()
    
    n = len(df)
    if n < 10:
        # Return NaN metrics for insufficient data
        return PredictionMetrics(
            oos_r2=np.nan, ic=np.nan, ic_pvalue=np.nan, hit_rate=np.nan,
            coverage=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan,
            interval_width=np.nan, ic_tstat=np.nan,
            n_observations=n, nominal_coverage=nominal_coverage
        )
    
    actual_data = df['actual']
    predicted_data = df['predicted']
    lower_ci_data = df['lower_ci']
    upper_ci_data = df['upper_ci']
    
    # Primary metrics
    # OOS R2
    ss_res = ((actual_data - predicted_data) ** 2).sum()
    ss_tot = ((actual_data - actual_data.mean()) ** 2).sum()
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # IC
    ic, ic_pvalue = spearmanr(actual_data, predicted_data)
    
    # Hit Rate - Exclude cases where actual is exactly 0
    mask_nonzero = actual_data != 0
    if mask_nonzero.sum() > 0:
        hit_rate = (np.sign(predicted_data[mask_nonzero]) == np.sign(actual_data[mask_nonzero])).mean()
    else:
        hit_rate = 0.5
    
    # Coverage
    coverage = ((actual_data >= lower_ci_data) & (actual_data <= upper_ci_data)).mean()
    
    # Secondary metrics
    rmse = np.sqrt(((actual_data - predicted_data) ** 2).mean())
    mae = np.abs(actual_data - predicted_data).mean()
    bias = (predicted_data - actual_data).mean()
    interval_width = (upper_ci_data - lower_ci_data).mean()
    
    # IC t-stat
    if abs(ic) >= 1:
        ic_tstat = np.inf * np.sign(ic)
    else:
        ic_tstat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2)
    
    return PredictionMetrics(
        oos_r2=oos_r2,
        ic=ic,
        ic_pvalue=ic_pvalue,
        hit_rate=hit_rate,
        coverage=coverage,
        rmse=rmse,
        mae=mae,
        bias=bias,
        interval_width=interval_width,
        ic_tstat=ic_tstat,
        n_observations=n,
        nominal_coverage=nominal_coverage
    )


def get_quality_rating(metric_name: str, value: float, nominal_coverage: float = 0.90) -> Tuple[str, str]:
    """
    Get quality rating and CSS class for a metric value.
    
    Returns:
        Tuple of (quality_label, css_class)
    """
    if np.isnan(value):
        return ("N/A", "neutral")
    
    thresholds = {
        'oos_r2': [(0.05, 'Excellent', 'excellent'), (0.02, 'Good', 'good'), (0.0, 'Marginal', 'marginal'), (-np.inf, 'Poor', 'poor')],
        'ic': [(0.10, 'Excellent', 'excellent'), (0.05, 'Good', 'good'), (0.02, 'Marginal', 'marginal'), (-np.inf, 'Poor', 'poor')],
        'hit_rate': [(0.60, 'Excellent', 'excellent'), (0.55, 'Good', 'good'), (0.50, 'Marginal', 'marginal'), (-np.inf, 'Poor', 'poor')],
        'bias': [(0.01, 'Good', 'good'), (0.02, 'Marginal', 'marginal'), (np.inf, 'Poor', 'poor')],  # USES ABS
        'ic_tstat': [(2.0, 'Significant', 'excellent'), (1.5, 'Marginal', 'marginal'), (-np.inf, 'Not Significant', 'poor')]
    }
    
    if metric_name == 'coverage':
        # Coverage: compare to nominal
        deviation = abs(value - nominal_coverage)
        if deviation <= 0.05:
            return ("Good", "good")
        elif deviation <= 0.10:
            return ("Marginal", "marginal")
        else:
            return ("Poor", "poor")
    
    val_to_check = abs(value) if metric_name == 'bias' else value
    
    if metric_name not in thresholds or thresholds[metric_name] is None:
        return ("—", "neutral")
    
    for threshold, label, css_class in thresholds[metric_name]:
        if val_to_check >= threshold:
            return (label, css_class)
    
    return ("Poor", "poor")


def format_metric_value(metric_name: str, value: float) -> str:
    """Format metric value for display."""
    if np.isnan(value):
        return "—"
    
    if metric_name in ['oos_r2', 'hit_rate', 'coverage', 'rmse', 'mae', 'interval_width']:
        return f"{value:.1%}"
    elif metric_name == 'bias':
        return f"{value:+.2%}"
    elif metric_name in ['ic', 'ic_tstat']:
        return f"{value:.2f}"
    elif metric_name == 'ic_pvalue':
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"

def compute_rolling_ic(actual: pd.Series, predicted: pd.Series, window: int = 36) -> pd.Series:
    """Compute rolling Spearman IC."""
    # Ensure they are aligned and clean
    df = pd.DataFrame({'a': actual, 'p': predicted}).dropna()
    results = []
    indices = []
    
    if len(df) < window:
        return pd.Series(dtype=float)
        
    for i in range(len(df) - window + 1):
        window_df = df.iloc[i : i + window]
        ic, _ = spearmanr(window_df['a'], window_df['p'])
        results.append(ic)
        indices.append(window_df.index[-1])
    
    return pd.Series(results, index=indices)

def compute_calibration_data(actual: pd.Series, predicted: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Compute data for calibration plot."""
    df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    # Use qcut to get roughly equal sized bins
    try:
        df['decile'] = pd.qcut(df['predicted'], n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fallback if not enough unique values
        df['decile'] = pd.cut(df['predicted'], n_bins, labels=False)
        
    calibration = df.groupby('decile').agg({
        'predicted': 'mean',
        'actual': 'mean',
        'decile': 'count'
    }).rename(columns={'decile': 'count'})
    return calibration

def compute_quintile_analysis(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Compute realized returns per prediction quintile."""
    df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    try:
        df['quintile'] = pd.qcut(df['predicted'], 5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
    except ValueError:
        df['quintile'] = pd.cut(df['predicted'], 5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        
    quintile_returns = df.groupby('quintile')['actual'].agg(['mean', 'std', 'count'])
    quintile_returns.columns = ['mean_return', 'std_return', 'n_obs']
    return quintile_returns

def compute_ic_by_regime(actual: pd.Series, predicted: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """Compute IC for each market regime."""
    df = pd.DataFrame({'actual': actual, 'predicted': predicted, 'regime': regime})
    results = []
    for r in df['regime'].unique():
        mask = df['regime'] == r
        if mask.sum() >= 12:
            ic, _ = spearmanr(df.loc[mask, 'actual'], df.loc[mask, 'predicted'])
            results.append({'regime': r, 'ic': ic, 'n_obs': mask.sum()})
    return pd.DataFrame(results)
