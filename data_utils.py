import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os
from yahooquery import Ticker

try:
    import streamlit as st
    def cache_data_wrapper(func):
        return st.cache_data(ttl=86400, show_spinner="Fetching Data...") (func)
except ImportError:
    def cache_data_wrapper(func):
        return func

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


@cache_data_wrapper
def load_fred_md_data(file_path: str = '2025-11-MD.csv') -> pd.DataFrame:
    """Load and process FRED-MD data for specified macro variables."""
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        df_raw = pd.read_csv(file_path)
        
        # Determine if it's a standard FRED-MD (has sasdate column) or PIT matrix (index)
        if 'sasdate' in df_raw.columns:
            # Transformation row detection
            try:
                # McCracken standard: Row 1 (df index 0) has transform codes (1-7)
                first_row = df_raw.iloc[0, 1:]
                is_trans = all(pd.to_numeric(first_row, errors='coerce').fillna(0).between(1, 7))
                if is_trans:
                    df = df_raw.iloc[1:].copy()
                else:
                    df = df_raw.copy()
            except:
                df = df_raw.iloc[1:].copy()
                
            df['sasdate'] = pd.to_datetime(df['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=['sasdate']).set_index('sasdate')
        else:
            # Assume it's a PIT matrix with index as date
            df = df_raw.copy()
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'date'})
            
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=[date_col]).set_index(date_col)
        
        # Align to Month End to match Asset Data
        df.index = df.index + pd.offsets.MonthEnd(0)
        
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



@cache_data_wrapper
def load_hybrid_asset_data(start_date: str = '1960-01-01', macro_file: str = '2025-11-MD.csv') -> pd.DataFrame:
    """
    Load hybrid asset data: ETF 'Head' spliced onto Macro Proxy 'Tail'.
    Uses Ratio Splicing to eliminate tracking error/look-ahead bias.
    """
    # 1. Fetch Proxy Tails (FRED-MD)
    # We need the raw series for backfilling
    df_proxies_raw = pd.DataFrame()
    try:
        if not os.path.exists(macro_file):
            vintage_dir = 'data/vintages'
            if os.path.exists(vintage_dir):
                v_files = sorted([f for f in os.listdir(vintage_dir) if f.endswith('.csv')])
                if v_files:
                    macro_file = os.path.join(vintage_dir, v_files[-1])

        if os.path.exists(macro_file):
            df_m = pd.read_csv(macro_file)
            if 'sasdate' in df_m.columns:
                df_m = df_m.iloc[1:] # Skip transform row
                df_m['date'] = pd.to_datetime(df_m['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
            else:
                date_col = 'Unnamed: 0' if 'Unnamed: 0' in df_m.columns else df_m.columns[0]
                df_m['date'] = pd.to_datetime(df_m[date_col], utc=True, errors='coerce').dt.tz_localize(None)
            
            # Align to Month End
            df_m['date'] = df_m['date'] + pd.offsets.MonthEnd(0)
            df_m = df_m.dropna(subset=['date']).set_index('date')
            
            # Map FRED-MD columns to our asset names
            sp_col = next((c for c in ['S&P 500', 'SP500', 'S&P_500'] if c in df_m.columns), None)
            if sp_col:
                df_proxies_raw['EQUITY'] = pd.to_numeric(df_m[sp_col], errors='coerce')
    except Exception as e:
        print(f"Error fetching Equity Proxy: {e}")

    # Gold Proxy (PPI)
    try:
        gold_ppi = web.DataReader('WPU1022', 'fred', start_date)
        gold_ppi.index = pd.to_datetime(gold_ppi.index, utc=True).tz_localize(None)
        gold_ppi = gold_ppi.resample('ME').last()
        df_proxies_raw['GOLD'] = gold_ppi['WPU1022']
    except Exception as e:
        print(f"Error fetching Gold Proxy: {e}")

    # Bond Proxy (Synthetic from GS10)
    try:
        gs10 = web.DataReader('GS10', 'fred', start_date)
        gs10.index = pd.to_datetime(gs10.index, utc=True).tz_localize(None)
        gs10 = gs10.resample('ME').last()
        yields = gs10['GS10'] / 100
        duration = 7.5
        carry = yields.shift(1) / 12
        price_change = -duration * (yields - yields.shift(1))
        synth_ret = carry + price_change
        # Use simple return for backcasting, but we'll return an index
        df_proxies_raw['BONDS_RET'] = synth_ret
    except Exception as e:
        print(f"Error fetching Bond Proxy: {e}")

    # 2. Fetch ETF Heads (YahooQuery)
    etf_map = {'SPY': 'EQUITY', 'IEF': 'BONDS', 'GLD': 'GOLD'}
    df_etfs = pd.DataFrame()
    try:
        # FIX: asynchronous=False to prevent event loop issues on Streamlit Cloud
        t = Ticker(list(etf_map.keys()), asynchronous=False)
        df_etf_raw = t.history(period='max', interval='1d')
        if not df_etf_raw.empty:
            df_etfs = df_etf_raw.reset_index().pivot(index='date', columns='symbol', values='adjclose')
            df_etfs = df_etfs.rename(columns=etf_map)
            df_etfs.index = pd.to_datetime(df_etfs.index, utc=True).tz_localize(None)
            df_etfs = df_etfs.resample('ME').last()
    except Exception as e:
        # FIX: Remove recursive call to load_asset_data(). 
        # Just log warning; Splice Engine will use proxies as fallback.
        if 'st' in globals():
            st.warning(f"Yahoo API failed: {e}. Using proxy data.")
        else:
            print(f"Error fetching ETF data: {e}")

    # 3. Splice Engine (Ratio Splicing)
    spliced_data = pd.DataFrame()
    
    for asset in ['EQUITY', 'BONDS', 'GOLD']:
        # T_splice = first valid date of ETF
        if asset in df_etfs.columns and not df_etfs[asset].dropna().empty:
            head_series = df_etfs[asset].dropna()
            t_splice = head_series.index[0]
            head = head_series.loc[t_splice:]
        else:
            # Fallback: No ETF data, use proxies for entire history
            t_splice = pd.Timestamp.max
            head = pd.Series(dtype=float)
            
        # Get Proxy Returns
        if asset == 'BONDS':
            # We already calculated returns for Bonds
            proxy_ret = df_proxies_raw.get('BONDS_RET', pd.Series(dtype=float))
        else:
            # Calculate % change for Equity and Gold proxies
            proxy_ret = df_proxies_raw[asset].pct_change() if asset in df_proxies_raw.columns else pd.Series(dtype=float)
            
        # Tail returns restricted to pre-inception
        tail_returns = proxy_ret.loc[:t_splice].iloc[::-1] # Reverse order for backcasting
        
        # Backcast Loop
        if not head.empty:
            current_price = head.iloc[0]
            history = []
            for date, ret in tail_returns.items():
                if date >= t_splice:
                    continue 
                if pd.isna(ret):
                    continue
                prev_price = current_price / (1 + ret)
                history.append((date, prev_price))
                current_price = prev_price
        else:
            # Full proxy reconstruction if no head exists
            # We start from 1.0 at the end of proxy data or fixed point
            current_price = 1.0
            history = []
            # For full proxy, we go forward then maybe normalize, 
            # but here it's easier to just use the raw proxy if it's EQUITY/GOLD
            if asset in df_proxies_raw.columns and asset != 'BONDS':
                spliced_data[asset] = df_proxies_raw[asset]
                continue
            elif asset == 'BONDS' and 'BONDS_RET' in df_proxies_raw.columns:
                # Reconstruct bond index from returns
                reconstructed = (1 + df_proxies_raw['BONDS_RET'].fillna(0)).cumprod()
                spliced_data[asset] = reconstructed
                continue
            history = []
        
        # Merge
        if history:
            tail = pd.DataFrame(history, columns=['date', asset]).set_index('date').sort_index()
            spliced_data[asset] = pd.concat([tail[asset], head])
        elif not head.empty:
            spliced_data[asset] = head

    # 4. Final Sanitization (Critical for preventing White Screen)
    # Replace 0 with NaN to avoid division by zero in returns, then drop all NaNs
    spliced_data = spliced_data.replace(0, np.nan).dropna()
    
    # Ensure strictly numeric and clean
    for col in spliced_data.columns:
        spliced_data[col] = pd.to_numeric(spliced_data[col], errors='coerce')
    
    return spliced_data.dropna()


def load_asset_data(start_date: str = '1960-01-01', macro_file: str = '2025-11-MD.csv') -> pd.DataFrame:
    """
    Deprecated: Wrapper for load_hybrid_asset_data to maintain backward compatibility.
    """
    return load_hybrid_asset_data(start_date, macro_file)
