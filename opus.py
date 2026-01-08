"""
High-Dimensional Adaptive Sparse Elastic Net VECM
Strategic Asset Allocation System - V2 (Fixed Stability + Long-Term Trends)

Target: US Equities / Bonds / Gold
Horizon: 5-10 years

FIXES:
- Stability analyzer now works with proper alpha scaling
- Trend analyzer uses 2Y vs 5Y (and 10Y) horizons for strategic view
- Better debugging output
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from yahooquery import Ticker
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas_datareader.data as web
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VECM Strategic Allocation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-tertiary: #1a1a1a;
        --border-color: #2a2a2a;
        --text-primary: #e8e8e8;
        --text-secondary: #888888;
        --text-muted: #555555;
        --accent-orange: #ff6b35;
        --accent-green: #00d26a;
        --accent-red: #ff4757;
        --accent-blue: #4da6ff;
        --accent-gold: #ffd700;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    .header-container {
        background: linear-gradient(180deg, #111111 0%, #0a0a0a 100%);
        border-bottom: 1px solid var(--border-color);
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-orange);
        letter-spacing: 0.5px;
        margin: 0;
    }
    
    .header-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    .panel-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-card {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0.25rem;
    }
    
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.warning { color: var(--accent-orange); }
    
    .data-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        width: 100%;
        border-collapse: collapse;
    }
    
    .data-table th {
        background-color: var(--bg-tertiary);
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        text-align: left;
    }
    
    .data-table td {
        color: var(--text-primary);
        padding: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton > button {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-orange);
        border-color: var(--accent-orange);
        color: #000;
    }
    
    .stTabs [data-baseweb="tab-list"] { background-color: var(--bg-secondary); gap: 0; }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-primary);
        color: var(--accent-orange);
        border-bottom-color: var(--bg-primary);
    }
    
    .debug-box {
        background: #1a1a1a;
        border: 1px solid #333;
        padding: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #888;
        margin: 0.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600)
def get_series_descriptions(file_path: str = 'FRED-MD_updated_appendix.csv') -> dict:
    """Load series descriptions from appendix."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        # Create mapping from fred ID to description
        mapping = dict(zip(df['fred'], df['description']))
        # Add some manual mappings for derived variables if any
        mapping['SPREAD'] = '10Y Treasury - Fed Funds Spread'
        mapping['BAA_AAA'] = 'Baa - Aaa Corporate Bond Spread'
        mapping['CAPACITY'] = 'Capacity Utilization: Manufacturing'
        return mapping
    except Exception as e:
        st.warning(f"Could not load series descriptions: {e}")
        return {}


@st.cache_data(ttl=3600)
def load_fred_md_data(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data."""
    try:
        df_raw = pd.read_csv(file_path)
        transform_codes = df_raw.iloc[0]
        df = df_raw.iloc[1:].copy()
        df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True).dt.tz_localize(None)
        df = df.set_index('sasdate')
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        data = pd.DataFrame(index=df.index)
        
        # Labor Market
        data['PAYEMS'] = df['PAYEMS']
        data['UNRATE'] = df['UNRATE']
        
        # Output & Production
        data['INDPRO'] = df['INDPRO']
        data['CAPACITY'] = df['CUMFNS']
        
        # Prices
        data['CPI'] = df['CPIAUCSL']
        data['PPI'] = df['WPSFD49207']
        data['PCE'] = df['PCEPI']
        
        # Financial Rates
        data['FEDFUNDS'] = df['FEDFUNDS']
        data['GS10'] = df['GS10']
        data['SPREAD'] = df['GS10'] - df['FEDFUNDS']
        
        # Credit & Housing
        data['BAA_AAA'] = df['BAA'] - df['AAA']
        data['HOUST'] = df['HOUST']
        
        # Money Supply
        data['M2'] = df['M2SL']
        
        # Apply log transformations where appropriate
        log_vars = ['PAYEMS', 'INDPRO', 'CPI', 'PPI', 'PCE', 'HOUST', 'M2']
        for col in log_vars:
            if col in data.columns:
                data[col] = np.log(data[col].replace(0, np.nan))
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data
        
    except Exception as e:
        st.error(f"Error loading FRED-MD data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_asset_data(start_date: str = '1960-01-01') -> pd.DataFrame:
    """Load long history asset data."""
    
    # EQUITIES from FRED-MD
    equity_ret = pd.Series(dtype=float)
    try:
        df_macro_raw = pd.read_csv('2025-11-MD.csv').iloc[1:]
        df_macro_raw['sasdate'] = pd.to_datetime(df_macro_raw['sasdate'], utc=True).dt.tz_localize(None)
        df_macro_raw.set_index('sasdate', inplace=True)
        
        if 'S&P 500' in df_macro_raw.columns:
            equity_data = pd.to_numeric(df_macro_raw['S&P 500'], errors='coerce').dropna()
            equity_ret = np.log(equity_data).diff().dropna()
    except Exception as e:
        st.warning(f"Equity data error: {e}")

    # GOLD - simplified
    gold_ret = pd.Series(dtype=float)
    try:
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        gold_ppi = gold_ppi.resample('MS').last()
        gold_ret = np.log(gold_ppi).diff().dropna()['WPU1022']
    except Exception as e:
        st.warning(f"Gold data error: {e}")

    # BONDS - synthetic from GS10
    bond_ret = pd.Series(dtype=float)
    try:
        gs10 = web.DataReader('GS10', 'fred', start_date)
        gs10.index = pd.to_datetime(gs10.index, utc=True).tz_localize(None)
        yields = gs10['GS10'] / 100
        duration = 7.5
        carry = yields.shift(1) / 12
        price_change = -duration * (yields - yields.shift(1))
        synth_ret = carry + price_change
        bond_ret = np.log(1 + synth_ret).dropna()
    except Exception as e:
        st.warning(f"Bond data error: {e}")

    all_ret = pd.DataFrame({
        'EQUITY': equity_ret,
        'GOLD': gold_ret,
        'BONDS': bond_ret
    }).dropna()
    
    df_prices = 100 * np.exp(all_ret.cumsum())
    return df_prices


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class KernelDictionary:
    """Simplified kernel dictionary for strategic horizon."""
    
    def __init__(self):
        # For 5-10Y horizon, focus on longer lags
        self.dense_lags = [1, 3, 6]  # Quarterly signal
        self.anchor_lags = [12, 24, 36, 60]  # 1Y to 5Y cycles
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features = {}
        
        # Dense lags
        for lag in self.dense_lags:
            for col in data.columns:
                features[f'{col}_L{lag}'] = data[col].shift(lag)
        
        # Moving averages for anchors (simpler than Gaussian)
        for anchor in self.anchor_lags:
            for col in data.columns:
                features[f'{col}_MA{anchor}'] = data[col].rolling(anchor).mean()
        
        result = pd.DataFrame(features, index=data.index)
        # Keep more data by forward-filling then dropping remaining NaN
        result = result.dropna()
        return result


class StabilityAnalyzer:
    """
    Fixed stability analyzer - uses expanding windows and lower alpha.
    """
    
    def __init__(self, n_windows: int = 5, min_persistence: float = 0.4):
        self.n_windows = n_windows
        self.min_persistence = min_persistence
        
    def analyze(self, y: pd.DataFrame, X: pd.DataFrame, 
                l1_ratio: float = 0.5, alpha: float = 0.001) -> dict:
        """
        Run stability analysis with proper scaling.
        
        Key fix: Use much lower alpha (0.001) because we're working with
        standardized data and monthly returns (small numbers).
        """
        common_idx = y.index.intersection(X.index)
        y_aligned = y.loc[common_idx].copy()
        X_aligned = X.loc[common_idx].copy()
        
        n_obs = len(common_idx)
        
        # Standardize X once
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_aligned),
            index=X_aligned.index,
            columns=X_aligned.columns
        )
        
        # Use expanding windows (more stable than rolling)
        # Window sizes: 50%, 60%, 70%, 80%, 90%, 100% of data
        window_fracs = np.linspace(0.5, 1.0, self.n_windows)
        
        all_coefficients = defaultdict(list)
        debug_info = {'n_obs': n_obs, 'n_features': len(X_aligned.columns), 'windows': []}
        
        for asset in y_aligned.columns:
            for frac in window_fracs:
                end_idx = int(n_obs * frac)
                if end_idx < 60:  # Minimum 60 observations
                    continue
                
                y_window = y_aligned.iloc[:end_idx][asset]
                X_window = X_scaled.iloc[:end_idx]
                
                # Use lower alpha - this is critical
                model = ElasticNet(
                    l1_ratio=l1_ratio,
                    alpha=alpha,  # Much lower than before
                    fit_intercept=True,
                    max_iter=10000,
                    tol=1e-5,
                    warm_start=False
                )
                
                try:
                    model.fit(X_window, y_window)
                    coefs = pd.Series(model.coef_, index=X_window.columns)
                    all_coefficients[asset].append(coefs)
                    
                    n_nonzero = (coefs != 0).sum()
                    debug_info['windows'].append({
                        'asset': asset,
                        'frac': frac,
                        'n_obs': end_idx,
                        'n_nonzero': n_nonzero
                    })
                except Exception as e:
                    debug_info['windows'].append({
                        'asset': asset,
                        'frac': frac,
                        'error': str(e)
                    })
        
        # Compute stability metrics
        results = {}
        
        for asset, coef_list in all_coefficients.items():
            if not coef_list:
                results[asset] = {'stable_features': [], 'debug': 'No coefficients estimated'}
                continue
            
            coef_df = pd.DataFrame(coef_list)
            
            # Persistence: how often is coefficient non-zero
            persistence = (coef_df.abs() > 1e-6).mean()
            
            # Sign consistency
            signs = np.sign(coef_df)
            sign_mode = signs.mode().iloc[0] if len(signs) > 0 else signs.iloc[0]
            sign_consistency = (signs == sign_mode).mean()
            
            # Mean coefficient (for ranking)
            mean_coef = coef_df.mean()
            
            # Combined score
            stability_score = persistence * sign_consistency
            
            # Select features that pass threshold
            # More lenient: just need to appear in min_persistence fraction of windows
            stable_mask = persistence >= self.min_persistence
            stable_features = stability_score[stable_mask].sort_values(ascending=False)
            
            # If still empty, take top 10 by absolute mean coefficient
            if len(stable_features) == 0:
                stable_features = mean_coef.abs().sort_values(ascending=False).head(10)
            
            results[asset] = {
                'stable_features': stable_features.head(15).index.tolist(),
                'stability_scores': stability_score,
                'persistence': persistence,
                'sign_consistency': sign_consistency,
                'mean_coefficients': mean_coef,
                'all_coefficients': coef_df,
                'n_windows': len(coef_list),
                'n_nonzero_avg': (coef_df.abs() > 1e-6).sum(axis=1).mean()
            }
        
        results['_debug'] = debug_info
        return results


class LongTermTrendAnalyzer:
    """
    Analyzes secular trends for 5-10 year investment horizon.
    Uses 2Y vs 5Y vs 10Y comparisons.
    """
    
    def __init__(self):
        self.windows = {
            'short': 24,   # 2 years
            'medium': 60,  # 5 years
            'long': 120    # 10 years
        }
    
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute long-term trends for each variable."""
        results = []
        
        for col in data.columns:
            series = data[col].dropna()
            n = len(series)
            
            if n < self.windows['medium']:
                continue
            
            current = series.iloc[-1]
            
            # 2-year trend
            if n >= self.windows['short']:
                ma_2y = series.rolling(self.windows['short']).mean()
                slope_2y = (ma_2y.iloc[-1] - ma_2y.iloc[-self.windows['short']]) / self.windows['short']
            else:
                slope_2y = 0
            
            # 5-year trend
            if n >= self.windows['medium']:
                ma_5y = series.rolling(self.windows['medium']).mean()
                slope_5y = (ma_5y.iloc[-1] - ma_5y.iloc[-self.windows['medium']]) / self.windows['medium']
            else:
                slope_5y = 0
            
            # 10-year trend (if available)
            if n >= self.windows['long']:
                ma_10y = series.rolling(self.windows['long']).mean()
                slope_10y = (ma_10y.iloc[-1] - ma_10y.iloc[-self.windows['long']]) / self.windows['long']
            else:
                slope_10y = np.nan
            
            # Normalize slopes by series std
            std = series.std()
            if std > 0:
                slope_2y_norm = slope_2y / std * 12  # Annualized
                slope_5y_norm = slope_5y / std * 12
                slope_10y_norm = slope_10y / std * 12 if not np.isnan(slope_10y) else np.nan
            else:
                slope_2y_norm = slope_5y_norm = slope_10y_norm = 0
            
            # Secular trend classification (based on 5Y trend)
            if slope_5y_norm > 0.1:
                secular_trend = 'SECULAR BULL'
            elif slope_5y_norm < -0.1:
                secular_trend = 'SECULAR BEAR'
            else:
                secular_trend = 'RANGE-BOUND'
            
            # Cycle position (2Y vs 5Y)
            if slope_2y_norm > slope_5y_norm + 0.05:
                cycle_position = 'ACCELERATING'
            elif slope_2y_norm < slope_5y_norm - 0.05:
                cycle_position = 'DECELERATING'
            else:
                cycle_position = 'ON TREND'
            
            # Historical percentile (where is current value vs history)
            percentile = (series < current).sum() / n * 100
            
            # Z-score vs 5Y average
            if n >= self.windows['medium']:
                avg_5y = series.iloc[-self.windows['medium']:].mean()
                std_5y = series.iloc[-self.windows['medium']:].std()
                z_score = (current - avg_5y) / std_5y if std_5y > 0 else 0
            else:
                z_score = 0
            
            results.append({
                'Variable': col,
                'Current': current,
                'Slope_2Y': slope_2y_norm,
                'Slope_5Y': slope_5y_norm,
                'Slope_10Y': slope_10y_norm,
                'Secular_Trend': secular_trend,
                'Cycle_Position': cycle_position,
                'Z_Score_5Y': z_score,
                'Percentile': percentile
            })
        
        return pd.DataFrame(results)


class RegimeSentinel:
    """Regime detection for risk management."""
    
    def __init__(self, alert_threshold: float = 2.0):
        self.alert_threshold = alert_threshold
        
    def evaluate(self, data: pd.DataFrame) -> tuple:
        recent = data.tail(60)
        
        indicators = {}
        
        # Credit stress
        if 'BAA_AAA' in recent.columns:
            spread_z = (recent['BAA_AAA'].iloc[-1] - recent['BAA_AAA'].mean()) / recent['BAA_AAA'].std()
            indicators['credit'] = spread_z
        else:
            indicators['credit'] = 0
        
        # Yield curve
        if 'SPREAD' in recent.columns:
            curve_z = -(recent['SPREAD'].iloc[-1] - recent['SPREAD'].mean()) / recent['SPREAD'].std()
            indicators['curve'] = curve_z
        else:
            indicators['curve'] = 0
        
        # Production
        if 'INDPRO' in recent.columns:
            prod_chg = recent['INDPRO'].pct_change(12).iloc[-1]
            indicators['production'] = -prod_chg / 0.03
        else:
            indicators['production'] = 0
        
        # Unemployment
        if 'UNRATE' in recent.columns:
            unrate_chg = recent['UNRATE'].iloc[-1] - recent['UNRATE'].iloc[-12]
            indicators['unemployment'] = unrate_chg / 0.5
        else:
            indicators['unemployment'] = 0
        
        # Composite score
        stress_score = (
            0.30 * indicators['credit'] +
            0.25 * indicators['curve'] +
            0.25 * indicators['production'] +
            0.20 * indicators['unemployment']
        )
        
        if stress_score > self.alert_threshold:
            status = "ALERT"
        elif stress_score > self.alert_threshold * 0.6:
            status = "WARNING"
        else:
            status = "CALM"
        
        return status, stress_score, indicators


class VECM:
    """Simplified VECM for cointegration analysis."""
    
    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha
        self.beta = None
        self.cointegration_vars = None
        
    def estimate_cointegration(self, levels):
        df = levels.copy()
        if 'SPREAD' in df.columns:
            df = df.drop(columns=['SPREAD'])
        
        self.cointegration_vars = list(df.columns)[:10]
        df = df[self.cointegration_vars]
        
        try:
            result = coint_johansen(df.values, det_order=0, k_ar_diff=2)
            self.beta = result.evec[:, 0]
            
            return {
                'rank': 1,
                'eigenvalues': result.eig,
                'trace_stats': result.lr1,
                'critical_values': result.cvt[:, 1]
            }
        except Exception as e:
            self.beta = np.ones(len(self.cointegration_vars)) / len(self.cointegration_vars)
            return {'rank': 1, 'error': str(e)}
    
    def compute_ect(self, levels) -> pd.Series:
        if self.beta is None or self.cointegration_vars is None:
            return pd.Series(0, index=levels.index)
        
        cols = [c for c in self.cointegration_vars if c in levels.columns]
        vals = levels[cols].values
        beta = self.beta[:len(cols)]
        
        ect = np.dot(vals, beta)
        return pd.Series(ect, index=levels.index, name='ECT')


def compute_strategic_signals(stability_results: dict, trend_analysis: pd.DataFrame, 
                              macro_cols: list) -> dict:
    """Compute net signals based on stable drivers and trends."""
    assets = ['EQUITY', 'BONDS', 'GOLD']
    signals = {}
    
    for asset in assets:
        if asset not in stability_results:
            continue
            
        result = stability_results[asset]
        stable_feats = result.get('stable_features', [])
        persistence = result.get('persistence', pd.Series())
        sign_cons = result.get('sign_consistency', pd.Series())
        mean_coefs = result.get('mean_coefficients', pd.Series())
        
        # 1. Filter high-confidence drivers (≥80% persistence and sign consistency)
        high_conf = [f for f in stable_feats if persistence.get(f, 0) >= 0.8 and sign_cons.get(f, 0) >= 0.8]
        
        driver_details = []
        total_signal = 0
        total_abs_signal = 0
        
        for feat in high_conf:
            # Extract base variable using prefix matching against macro columns
            base_var = None
            for col in macro_cols:
                if feat == col or feat.startswith(col + '_'):
                    base_var = col
                    break
            if not base_var:
                base_var = feat.split('_')[0]
                
            coef = mean_coefs.get(feat, 0)
            pers = persistence.get(feat, 0)
            sign_c = sign_cons.get(feat, 0)
            
            # Get trend info
            trend_row = trend_analysis[trend_analysis['Variable'] == base_var]
            if not trend_row.empty:
                slope_5y = trend_row['Slope_5Y'].iloc[0]
                secular = trend_row['Secular_Trend'].iloc[0]
            else:
                slope_5y = 0
                secular = 'N/A'
            
            # 3. Compute signal contribution
            weight = pers * sign_c
            signal_contrib = coef * slope_5y * weight
            
            # 4. Classify effect
            if coef > 0:
                effect = 'BULLISH' if slope_5y > 0.05 else 'BEARISH' if slope_5y < -0.05 else 'NEUTRAL'
            else:
                effect = 'BEARISH' if slope_5y > 0.05 else 'BULLISH' if slope_5y < -0.05 else 'NEUTRAL'
                
            driver_details.append({
                'Driver': feat,
                'Coefficient': coef,
                '5Y Trend': secular,
                '5Y Slope': slope_5y,
                'Signal': effect,
                'Contribution': signal_contrib
            })
            
            total_signal += signal_contrib
            total_abs_signal += abs(signal_contrib)
            
        # 5. Aggregate net signal (Normalized to range [-1, +1])
        net_signal = total_signal / total_abs_signal if total_abs_signal > 0 else 0
        
        # 6. Signal-based adjustment classification
        if net_signal > 0.3:
            rec = 'OVERWEIGHT'
        elif net_signal < -0.3:
            rec = 'UNDERWEIGHT'
        else:
            rec = 'NEUTRAL'
            
        signals[asset] = {
            'net_signal': net_signal,
            'recommendation': rec,
            'n_drivers': len(high_conf),
            'bullish': sum(1 for d in driver_details if d['Signal'] == 'BULLISH'),
            'bearish': sum(1 for d in driver_details if d['Signal'] == 'BEARISH'),
            'neutral': sum(1 for d in driver_details if d['Signal'] == 'NEUTRAL'),
            'details': driver_details
        }
    
    return signals


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_theme():
    return {
        'paper_bgcolor': '#0a0a0a',
        'plot_bgcolor': '#111111',
        'font': {'family': 'IBM Plex Mono', 'color': '#888888', 'size': 11},
        'xaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a'},
        'yaxis': {'gridcolor': '#1a1a1a', 'linecolor': '#2a2a2a'}
    }


def plot_allocation(weights: dict) -> go.Figure:
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.65,
        marker=dict(colors=[colors[k] for k in weights.keys()], line=dict(color='#0a0a0a', width=2)),
        textinfo='label+percent',
        textfont=dict(family='IBM Plex Mono', size=11, color='#e8e8e8')
    )])
    
    theme = create_theme()
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
        annotations=[dict(text='<b>TARGET</b>', x=0.5, y=0.5,
                         font=dict(family='IBM Plex Mono', size=12, color='#888'), showarrow=False)]
    )
    return fig


def plot_assets(prices: pd.DataFrame) -> go.Figure:
    theme = create_theme()
    normalized = prices / prices.iloc[0] * 100
    
    fig = go.Figure()
    colors = {'EQUITY': '#ff6b35', 'BONDS': '#4da6ff', 'GOLD': '#ffd700'}
    
    for col in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col], name=col,
            mode='lines', line=dict(color=colors.get(col, '#888'), width=1.5)
        ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=30, b=40),
        height=280,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', title='Indexed (100)')
    )
    return fig


def plot_ect(ect: pd.Series) -> go.Figure:
    theme = create_theme()
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#2a2a2a")
    
    std = ect.std()
    fig.add_hrect(y0=-std, y1=std, fillcolor="rgba(77, 166, 255, 0.05)", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=ect.index, y=ect.values, mode='lines',
        line=dict(color='#4da6ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(77, 166, 255, 0.1)'
    ))
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=20, t=10, b=40),
        height=200,
        xaxis=dict(gridcolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', title='ECT')
    )
    return fig


def plot_stability_boxplot(results: dict, asset: str, descriptions: dict = None) -> go.Figure:
    """Show coefficient distribution across windows."""
    theme = create_theme()
    
    if asset not in results or 'all_coefficients' not in results[asset]:
        return go.Figure()
    
    coef_df = results[asset]['all_coefficients']
    stable_feats = results[asset].get('stable_features', [])[:8]
    
    if not stable_feats:
        return go.Figure()
    
    fig = go.Figure()
    colors = ['#ff6b35', '#4da6ff', '#ffd700', '#00d26a', '#ff4757', '#9b59b6', '#1abc9c', '#e74c3c']
    
    for i, feat in enumerate(stable_feats):
        if feat in coef_df.columns:
            base_var = feat.split('_')[0]
            desc = descriptions.get(base_var, base_var) if descriptions else base_var
            
            fig.add_trace(go.Box(
                y=coef_df[feat], 
                name=feat[:12],
                marker_color=colors[i % len(colors)],
                boxmean=True,
                hovertext=desc,
                hoverinfo='y+text+name'
            ))
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        showlegend=False,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#1a1a1a', title='Coefficient')
    )
    return fig


def plot_trend_bars(trend_df: pd.DataFrame, variables: list, descriptions: dict = None) -> go.Figure:
    """Plot 5Y trend comparison."""
    theme = create_theme()
    
    filtered = trend_df[trend_df['Variable'].isin(variables)]
    if filtered.empty:
        return go.Figure()
    
    colors = ['#00d26a' if x > 0.05 else '#ff4757' if x < -0.05 else '#888888' 
              for x in filtered['Slope_5Y']]
    
    hovers = []
    for var in filtered['Variable']:
        desc = descriptions.get(var, var) if descriptions else var
        hovers.append(desc)
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered['Variable'],
        y=filtered['Slope_5Y'],
        marker_color=colors,
        text=[f"{x:.2f}" for x in filtered['Slope_5Y']],
        textposition='outside',
        hovertext=hovers,
        hoverinfo='y+text+x'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#2a2a2a")
    
    fig.update_layout(
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=40, r=20, t=20, b=60),
        height=250,
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='#1a1a1a', title='5Y Slope (Annualized)')
    )
    return fig


def plot_driver_vs_asset(feat_data: pd.DataFrame, asset_prices: pd.DataFrame, 
                         feat_name: str, asset: str, descriptions: dict = None) -> go.Figure:
    """Plot dual-axis comparison of macro driver and asset price with normalization."""
    theme = create_theme()
    
    if feat_name not in feat_data.columns or asset not in asset_prices.columns:
        return go.Figure()
        
    # Align data
    combined = pd.concat([feat_data[feat_name], asset_prices[asset]], axis=1).dropna()
    if combined.empty:
        return go.Figure()
        
    # Normalize for visual comparison: (x - min) / (max - min)
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) != 0 else s
    
    macro_norm = normalize(combined[feat_name])
    # Log-transform asset price to reveal early history dynamics
    asset_norm = normalize(np.log(combined[asset]))
    
    base_var = feat_name.split('_')[0]
    desc = descriptions.get(base_var, base_var) if descriptions else base_var
    title_text = f"{feat_name} ({desc}) vs {asset} (Log-Scaled)"
    
    fig = go.Figure()
    
    # Macro series (left axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=macro_norm, name=feat_name,
        mode='lines', line=dict(color='#00d26a', width=1.5),
        hovertemplate="<b>" + feat_name + "</b> (Norm): %{y:.2f}<extra></extra>"
    ))
    
    # Asset price (right axis)
    fig.add_trace(go.Scatter(
        x=combined.index, y=asset_norm, name=asset,
        mode='lines', line=dict(color='#4da6ff', width=1.5),
        yaxis='y2',
        hovertemplate="<b>" + asset + "</b> (Norm): %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title_text, 
                  font=dict(family='IBM Plex Mono', size=11, color='#888')),
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        margin=dict(l=50, r=50, t=40, b=40),
        height=350,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#1a1a1a',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dash',
            spikethickness=1,
            spikecolor='#555'
        ),
        yaxis=dict(gridcolor='#1a1a1a', side='left', title='Macro (Normalized)', range=[0, 1]),
        yaxis2=dict(gridcolor='#1a1a1a', overlaying='y', side='right', title='Asset (Normalized)', range=[0, 1])
    )
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("""
    <div class="header-container">
        <p class="header-title">◈ VECM STRATEGIC ALLOCATION</p>
        <p class="header-subtitle">5-10 YEAR HORIZON · STABILITY-ENHANCED · V2</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid #2a2a2a; margin-bottom: 1rem;">
            <span style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #ff6b35;">CONFIG</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Stability Analysis")
        
        n_windows = st.slider("Windows", 3, 8, 5, help="Number of expanding windows")
        min_persistence = st.slider("Min Persistence", 0.2, 0.8, 0.4, 0.1,
                                    help="Min fraction of windows where feature must appear")
        
        st.markdown("##### Elastic Net")
        
        l1_ratio = st.slider("L1 Ratio", 0.1, 0.9, 0.5, 0.1,
                             help="0=Ridge (grouped), 1=Lasso (sparse)")
        alpha = st.select_slider("Alpha (Regularization)", 
                                 options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                 value=0.001,
                                 help="Lower = more features selected")
        
        st.markdown("##### Risk")
        alert_threshold = st.slider("Alert Threshold", 1.0, 3.0, 2.0, 0.25)
        
        show_debug = st.checkbox("Show Debug Info", value=False)
    
    # Load data
    macro_data = load_fred_md_data()
    asset_prices = load_asset_data()
    descriptions = get_series_descriptions()
    
    if macro_data.empty:
        st.error("Failed to load macro data")
        return
    
    common_idx = macro_data.index.intersection(asset_prices.index)
    asset_prices = asset_prices.loc[common_idx]
    
    # Initialize components
    kernel = KernelDictionary()
    stability = StabilityAnalyzer(n_windows=n_windows, min_persistence=min_persistence)
    trend_analyzer = LongTermTrendAnalyzer()
    sentinel = RegimeSentinel(alert_threshold=alert_threshold)
    vecm = VECM(alpha=alpha)
    
    # Run analysis
    kernel_features = kernel.transform(macro_data)
    asset_returns = np.log(asset_prices).diff().dropna()
    
    # Stability analysis
    stability_results = stability.analyze(
        asset_returns, kernel_features,
        l1_ratio=l1_ratio, alpha=alpha
    )
    
    # Long-term trends
    trend_analysis = trend_analyzer.analyze(macro_data)
    
    # Regime
    status, stress_score, stress_indicators = sentinel.evaluate(macro_data)
    
    # Cointegration
    coint_results = vecm.estimate_cointegration(macro_data)
    ect = vecm.compute_ect(macro_data)
    
    # Generate allocation based on signal and regime
    base_allocs = {
        'ALERT': {'EQUITY': 0.30, 'BONDS': 0.45, 'GOLD': 0.25},
        'WARNING': {'EQUITY': 0.45, 'BONDS': 0.40, 'GOLD': 0.15},
        'CALM': {'EQUITY': 0.60, 'BONDS': 0.30, 'GOLD': 0.10}
    }
    
    current_base = base_allocs.get(status, base_allocs['CALM'])
    strategic_signals = compute_strategic_signals(stability_results, trend_analysis, list(macro_data.columns))
    
    # 0.15 max adjustment per asset based on net signal
    adjustment_factor = 0.15
    adjusted_weights = {}
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        net_sig = strategic_signals.get(asset, {}).get('net_signal', 0)
        adj = net_sig * adjustment_factor
        raw_weight = current_base[asset] + adj
        # Clamp between 5% and 85%
        adjusted_weights[asset] = max(0.05, min(0.85, raw_weight))
        
    # Final normalized weights
    total_w = sum(adjusted_weights.values())
    weights = {k: v / total_w for k, v in adjusted_weights.items()}
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = 'positive' if status == 'CALM' else 'warning' if status == 'WARNING' else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Regime</div>
            <div class="metric-value {color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = 'positive' if stress_score < 1.2 else 'warning' if stress_score < 2.0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stress</div>
            <div class="metric-value {color}">{stress_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        n_stable = sum(len(r.get('stable_features', [])) for k, r in stability_results.items() if k != '_debug')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stable Drivers</div>
            <div class="metric-value">{n_stable}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ect_val = ect.iloc[-1] if len(ect) > 0 else 0
        color = 'positive' if ect_val > 0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ECT</div>
            <div class="metric-value {color}">{ect_val:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Points</div>
            <div class="metric-value">{len(common_idx)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Debug info
    if show_debug and '_debug' in stability_results:
        debug = stability_results['_debug']
        st.markdown(f"""
        <div class="debug-box">
            <b>Debug:</b> n_obs={debug['n_obs']}, n_features={debug['n_features']}<br>
            Windows: {len(debug['windows'])} estimations<br>
            Sample: {debug['windows'][:3] if debug['windows'] else 'None'}
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["ALLOCATION", "STABLE DRIVERS", "SECULAR TRENDS", "DIAGNOSTICS"])
    
    with tab1:
        st.markdown('<div class="panel-header">STRATEGIC POSITIONING SUMMARY</div>', unsafe_allow_html=True)
        
        # Summary Table
        summary_rows = []
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            if asset in strategic_signals:
                sig = strategic_signals[asset]
                summary_rows.append({
                    'Asset': asset,
                    '# Drivers': sig['n_drivers'],
                    'Bullish': sig['bullish'],
                    'Bearish': sig['bearish'],
                    'Neutral': sig['neutral'],
                    'Net Signal': f"{sig['net_signal']:+.2f}",
                    'Recommendation': sig['recommendation']
                })
        
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            
            # Expandable Details
            cols = st.columns(3)
            for i, asset in enumerate(['EQUITY', 'BONDS', 'GOLD']):
                with cols[i]:
                    with st.expander(f"Drivers: {asset}"):
                        if asset in strategic_signals:
                            sig = strategic_signals[asset]
                            if sig['details']:
                                details_df = pd.DataFrame(sig['details'])
                                # Display selection
                                disp_df = details_df[['Driver', 'Signal', '5Y Trend', 'Contribution']].copy()
                                disp_df['Contribution'] = disp_df['Contribution'].map(lambda x: f"{x:+.5f}")
                                st.dataframe(disp_df, hide_index=True)
                            else:
                                st.caption("No high-confidence drivers.")
            
            # Narrative rationalein
            st.markdown('<div class="panel-header">STRATEGIC RATIONALE</div>', unsafe_allow_html=True)
            narratives = []
            for asset in ['EQUITY', 'BONDS', 'GOLD']:
                if asset in strategic_signals:
                    sig = strategic_signals[asset]
                    details = sig['details']
                    if details:
                        top_bullish = next((d['Driver'] for d in sorted(details, key=lambda x: x['Contribution'], reverse=True) if d['Signal'] == 'BULLISH'), None)
                        top_bearish = next((d['Driver'] for d in sorted(details, key=lambda x: x['Contribution']) if d['Signal'] == 'BEARISH'), None)
                        
                        nar = f"**{asset}**: **{sig['recommendation']}** ({sig['n_drivers']} drivers)."
                        if top_bullish: nar += f" {top_bullish} is a major tailwind."
                        if top_bearish: nar += f" {top_bearish} is a headwind."
                        narratives.append(nar)
            
            if narratives:
                st.markdown(f"""
                <div style="background:#111; border:1px solid #2a2a2a; padding:1rem; margin-bottom:2rem; border-radius:2px; font-size:0.85rem; color:#ccc; font-family:'IBM Plex Sans';">
                    {"<br>".join(narratives)}<br><br>
                    <span style="color:#888; font-size:0.75rem;">Regime: {status} | Stress Score: {stress_score:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data for strategic signal generation.")

        col_l, col_r = st.columns([1, 2])
        
        with col_l:
            st.markdown('<div class="panel-header">TARGET ALLOCATION</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_allocation(weights), use_container_width=True, config={'displayModeBar': False})
            
            st.markdown(f"""
            <table class="data-table">
                <tr><th>Asset</th><th>Weight</th></tr>
                <tr><td style="color:#ff6b35">EQUITY</td><td>{weights['EQUITY']:.0%}</td></tr>
                <tr><td style="color:#4da6ff">BONDS</td><td>{weights['BONDS']:.0%}</td></tr>
                <tr><td style="color:#ffd700">GOLD</td><td>{weights['GOLD']:.0%}</td></tr>
            </table>
            """, unsafe_allow_html=True)
        
        with col_r:
            st.markdown('<div class="panel-header">ASSET PERFORMANCE (10Y)</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_assets(asset_prices.tail(120)), use_container_width=True, config={'displayModeBar': False})
            
            st.markdown('<div class="panel-header">ERROR CORRECTION TERM</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_ect(ect.tail(240)), use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        st.markdown(f"""
        <div style="background:#111; border:1px solid #2a2a2a; padding:1rem; margin-bottom:1rem; border-radius:2px;">
            <span style="font-family:'IBM Plex Mono'; font-size:0.85rem; color:#888;">
                <span style="color:#ff6b35">STABILITY ANALYSIS:</span> 
                Features below appeared in ≥{min_persistence:.0%} of {n_windows} expanding windows.
                These are reliable macro drivers for strategic positioning.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            st.markdown(f'<div class="panel-header">{asset}</div>', unsafe_allow_html=True)
            
            if asset in stability_results:
                result = stability_results[asset]
                stable_feats = result.get('stable_features', [])
                persistence = result.get('persistence', pd.Series())
                sign_cons = result.get('sign_consistency', pd.Series())
                mean_coefs = result.get('mean_coefficients', pd.Series())
                
                if stable_feats:
                    rows = []
                    for feat in stable_feats[:10]:
                        base_var = feat.split('_')[0]
                        lag_type = '_'.join(feat.split('_')[1:]) if '_' in feat else 'L1'
                        
                        # Get trend
                        trend_row = trend_analysis[trend_analysis['Variable'] == base_var]
                        if not trend_row.empty:
                            secular = trend_row['Secular_Trend'].iloc[0]
                            slope_5y = trend_row['Slope_5Y'].iloc[0]
                        else:
                            secular = 'N/A'
                            slope_5y = 0
                        
                        coef = mean_coefs.get(feat, 0)
                        pers = persistence.get(feat, 0)
                        sign_c = sign_cons.get(feat, 0)
                        
                        rows.append({
                            'Feature': feat,
                            'Description': descriptions.get(base_var, 'N/A'),
                            'Base Var': base_var,
                            'Lag': lag_type,
                            'Coefficient': f"{coef:+.4f}",
                            'Persistence': f"{pers:.0%}",
                            'Sign Consist.': f"{sign_c:.0%}",
                            '5Y Trend': secular,
                            '5Y Slope': f"{slope_5y:+.2f}"
                        })
                    
                    df_key = f"df_{asset}"
                    selection = st.dataframe(
                        pd.DataFrame(rows), 
                        hide_index=True, 
                        use_container_width=True,
                        on_select='rerun',
                        selection_mode='single-row',
                        key=df_key
                    )
                    
                    # Handle selection
                    selected_rows = selection.get('selection', {}).get('rows', [])
                    if selected_rows:
                        row_idx = selected_rows[0]
                        selected_feat = rows[row_idx]['Feature']
                        
                        st.plotly_chart(
                            plot_driver_vs_asset(kernel_features, asset_prices, selected_feat, asset, descriptions),
                            use_container_width=True, 
                            config={'displayModeBar': False}
                        )
                    
                    # Boxplot
                    st.plotly_chart(plot_stability_boxplot(stability_results, asset, descriptions),
                                   use_container_width=True, config={'displayModeBar': False})
                else:
                    # Show what we have anyway
                    st.warning(f"No stable features found for {asset}. Showing top coefficients instead.")
                    if 'mean_coefficients' in result:
                        top_coefs = result['mean_coefficients'].abs().sort_values(ascending=False).head(10)
                        st.dataframe(pd.DataFrame({
                            'Feature': top_coefs.index,
                            'Abs Coefficient': top_coefs.values
                        }), hide_index=True)
            
            st.divider()
    
    with tab3:
        st.markdown('<div class="panel-header">SECULAR TRENDS (5-10 YEAR VIEW)</div>', unsafe_allow_html=True)
        
        # Format trend table
        display_df = trend_analysis[['Variable', 'Secular_Trend', 'Cycle_Position', 
                                     'Slope_2Y', 'Slope_5Y', 'Slope_10Y', 'Z_Score_5Y', 'Percentile']].copy()
        display_df['Description'] = display_df['Variable'].map(descriptions)
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Variable': st.column_config.TextColumn('Variable'),
                'Description': st.column_config.TextColumn('Description'),
                'Secular_Trend': st.column_config.TextColumn('5Y Trend'),
                'Cycle_Position': st.column_config.TextColumn('Cycle'),
                'Slope_2Y': st.column_config.NumberColumn('2Y Slope', format="%.2f"),
                'Slope_5Y': st.column_config.NumberColumn('5Y Slope', format="%.2f"),
                'Slope_10Y': st.column_config.NumberColumn('10Y Slope', format="%.2f"),
                'Z_Score_5Y': st.column_config.NumberColumn('Z-Score', format="%.2f"),
                'Percentile': st.column_config.NumberColumn('Pctl', format="%.0f%%")
            }
        )
        
        st.markdown('<div class="panel-header">KEY DRIVERS BY ASSET</div>', unsafe_allow_html=True)
        
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            if asset in stability_results:
                stable_feats = stability_results[asset].get('stable_features', [])[:8]
                base_vars = list(set([f.split('_')[0] for f in stable_feats]))
                
                if base_vars:
                    st.markdown(f"**{asset}** key drivers:")
                    st.plotly_chart(plot_trend_bars(trend_analysis, base_vars, descriptions),
                                   use_container_width=True, config={'displayModeBar': False})
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-header">COINTEGRATION</div>', unsafe_allow_html=True)
            if 'eigenvalues' in coint_results:
                coint_df = pd.DataFrame({
                    'r': range(len(coint_results['eigenvalues'])),
                    'Eigenvalue': coint_results['eigenvalues'],
                    'Trace': coint_results['trace_stats'],
                    'Crit 5%': coint_results['critical_values']
                })
                st.dataframe(coint_df.head(5), hide_index=True)
        
        with col2:
            st.markdown('<div class="panel-header">STRESS INDICATORS</div>', unsafe_allow_html=True)
            stress_df = pd.DataFrame([
                {'Indicator': k.title(), 'Z-Score': v, 'Status': 'OK' if abs(v) < 1.5 else 'ELEVATED'}
                for k, v in stress_indicators.items()
            ])
            st.dataframe(stress_df, hide_index=True)
        
        st.markdown('<div class="panel-header">PARAMETERS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Parameter | Value | Notes |
        |-----------|-------|-------|
        | L1 Ratio | {l1_ratio} | {'Ridge-biased' if l1_ratio < 0.4 else 'Balanced' if l1_ratio < 0.7 else 'Lasso-biased'} |
        | Alpha | {alpha} | {'Very light' if alpha < 0.001 else 'Light' if alpha < 0.005 else 'Moderate'} |
        | Windows | {n_windows} | Expanding windows |
        | Min Persistence | {min_persistence:.0%} | Feature selection threshold |
        | Data Range | {macro_data.index[0].strftime('%Y-%m')} to {macro_data.index[-1].strftime('%Y-%m')} | {len(macro_data)} months |
        """)
    
    # Footer
    st.markdown("""
    <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #2a2a2a; text-align:center;">
        <span style="font-family:'IBM Plex Mono'; font-size:0.65rem; color:#555;">
            VECM V2 · STABILITY-ENHANCED · 5-10Y HORIZON
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()