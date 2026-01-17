import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os

def compute_forward_returns(prices: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    Compute annualized forward returns for each asset.
    """
    log_prices = np.log(prices)
    forward_log_return = log_prices.shift(-horizon_months) - log_prices
    annualized_return = forward_log_return / (horizon_months / 12)
    return annualized_return


def prepare_macro_features(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform macro variables into current-state features.
    No forward-looking information used.
    """
    features = pd.DataFrame(index=macro_data.index)
    
    for col in macro_data.columns:
        series = macro_data[col]
        
        # Level
        features[f'{col}_level'] = series
        
        # Moving averages
        features[f'{col}_MA12'] = series.rolling(12).mean()
        features[f'{col}_MA60'] = series.rolling(60).mean()
        
        # Z-score vs 5Y average
        ma60 = series.rolling(60).mean()
        std60 = series.rolling(60).std()
        features[f'{col}_zscore'] = (series - ma60) / (std60 + 1e-9)
        
        # Percentile rank (rolling 10Y window)
        features[f'{col}_pctl'] = series.rolling(120).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
        )
        
        # Momentum / slope
        ma12 = series.rolling(12).mean()
        std = series.rolling(60).std()
        features[f'{col}_slope12'] = (ma12 - ma12.shift(12)) / (std + 1e-9)
        features[f'{col}_slope60'] = (ma60 - ma60.shift(60)) / (std + 1e-9)
    
    return features.dropna()


def load_fred_md_data(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data for specified macro variables."""
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        df_raw = pd.read_csv(file_path)
        # Drop the transformation row if it exists (usually row 0 in pandas if header is handled, 
        # but FRED-MD CSVs have row 1 as transformation codes)
        # df_raw.iloc[0] is the first data row if we use header=0. 
        # Let's check the logic used in app.py
        df = df_raw.iloc[1:].copy()
        df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True).dt.tz_localize(None)
        df = df.set_index('sasdate')
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Core Macro Variables from Spec
        mapping = {
            'PAYEMS': 'PAYEMS',     # Labor
            'UNRATE': 'UNRATE',     # Labor
            'INDPRO': 'INDPRO',     # Output
            'CUMFNS': 'CAPACITY',   # Output
            'CPIAUCSL': 'CPI',      # Prices
            'WPSFD49207': 'PPI',    # Prices
            'PCEPI': 'PCE',         # Prices
            'FEDFUNDS': 'FEDFUNDS', # Rates
            'GS10': 'GS10',         # Rates
            'HOUST': 'HOUST',       # Housing
            'M2SL': 'M2'            # Money
        }
        
        data = pd.DataFrame(index=df.index)
        for fred_col, target_col in mapping.items():
            if fred_col in df.columns:
                data[target_col] = df[fred_col]
        
        # Derived Financial Variables
        if 'GS10' in data.columns and 'FEDFUNDS' in data.columns:
            data['SPREAD'] = data['GS10'] - data['FEDFUNDS']
        
        if 'BAA' in df.columns and 'AAA' in df.columns:
            data['BAA_AAA'] = df['BAA'] - df['AAA']
            
        log_vars = ['PAYEMS', 'INDPRO', 'CPI', 'PPI', 'PCE', 'HOUST', 'M2']
        for col in log_vars:
            if col in data.columns:
                data[col] = np.log(data[col].replace(0, np.nan))
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data
        
    except Exception as e:
        print(f"Error loading FRED-MD data: {e}")
        return pd.DataFrame()


def load_asset_data(start_date: str = '1960-01-01', macro_file: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load long history asset prices."""
    
    # EQUITIES from FRED-MD (S&P 500)
    equity_prices = pd.Series(dtype=float)
    try:
        if os.path.exists(macro_file):
            df_macro_raw = pd.read_csv(macro_file).iloc[1:]
            df_macro_raw['sasdate'] = pd.to_datetime(df_macro_raw['sasdate'], utc=True).dt.tz_localize(None)
            df_macro_raw.set_index('sasdate', inplace=True)
            
            if 'S&P 500' in df_macro_raw.columns:
                equity_prices = pd.to_numeric(df_macro_raw['S&P 500'], errors='coerce').dropna()
    except Exception as e:
        print(f"Equity data error: {e}")

    # GOLD - using PPI for Gold (WPU1022) as long-term proxy
    gold_prices = pd.Series(dtype=float)
    try:
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        gold_ppi = gold_ppi.resample('MS').last()
        gold_prices = gold_ppi['WPU1022'].dropna()
    except Exception as e:
        print(f"Gold data error: {e}")

    # BONDS - synthetic total return from GS10
    bond_prices = pd.Series(dtype=float)
    try:
        gs10 = web.DataReader('GS10', 'fred', start_date)
        gs10.index = pd.to_datetime(gs10.index, utc=True).tz_localize(None)
        yields = gs10['GS10'] / 100
        duration = 7.5
        # Calculate monthly total returns
        carry = yields.shift(1) / 12
        price_change = -duration * (yields - yields.shift(1))
        synth_ret = carry + price_change
        # Convert returns to index
        bond_prices = 100 * (1 + synth_ret.fillna(0)).cumprod()
    except Exception as e:
        print(f"Bond data error: {e}")

    df_prices = pd.DataFrame({
        'EQUITY': equity_prices,
        'GOLD': gold_prices,
        'BONDS': bond_prices
    }).dropna()
    
    return df_prices
