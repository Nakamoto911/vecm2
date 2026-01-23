import pandas as pd
import numpy as np
from app import load_full_fred_md_raw, compute_forward_returns, load_asset_data

def debug_scales():
    print("DEBUG: Loading data...")
    asset_prices = load_asset_data()
    raw_macro, _ = load_full_fred_md_raw()
    
    # Check Asset Prices
    print("\n[Asset Prices Summary]")
    print(asset_prices.describe())
    
    # Check Raw Macro (specifically FEDFUNDS)
    print("\n[Raw Macro FEDFUNDS Summary]")
    if 'FEDFUNDS' in raw_macro.columns:
        print(raw_macro['FEDFUNDS'].describe())
    else:
        print("FEDFUNDS not found in raw_macro")

    # Compute y_nominal
    y_nominal = compute_forward_returns(asset_prices, horizon_months=12, vol_scale=False, excess_return=False)
    
    print("\n[y_nominal Summary (Decimal or Percent?)]")
    print(y_nominal.describe())
    
    # Check specific values around Oct 2021 (from user image)
    target_date = pd.Timestamp("2021-10-31")
    if target_date in y_nominal.index:
        print(f"\n[y_nominal at {target_date}]")
        print(y_nominal.loc[target_date])
    else:
        # Find closest date
        closest_idx = y_nominal.index.get_indexer([target_date], method='nearest')
        closest_date = y_nominal.index[closest_idx][0]
        print(f"\n[y_nominal closest to {target_date} is {closest_date}]")
        print(y_nominal.loc[closest_date])

if __name__ == "__main__":
    debug_scales()
