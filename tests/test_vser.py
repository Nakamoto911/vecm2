import pandas as pd
import numpy as np
from data_utils import compute_forward_returns

def test_vser_calculation():
    # Setup mock data
    dates = pd.date_range('2020-01-01', periods=60, freq='ME')
    prices = pd.DataFrame({'EQUITY': [100 * (1.01**i) for i in range(60)]}, index=dates)
    # Price is growing at 1% per month -> Annualized log return ~ 12%
    
    macro_data = pd.DataFrame({'FEDFUNDS': [4.0] * 60}, index=dates)
    
    # Calculate VSER
    # Horizon 12 months.
    # log(prices[12]) - log(prices[0]) = 12 * log(1.01) = 0.1194
    # Annualized = 0.1194 / 1.0 = 0.1194
    # Rf = 0.04
    # Excess = 0.1194 - 0.04 = 0.0794
    # Vol = monthly std * sqrt(12)
    # monthly returns = log(1.01) = 0.00995
    # std = 0 (since constant)
    # Wait, if std is 0, VSER might be inf. 
    # Let's add some noise to prices for vol calculation.
    
    np.random.seed(42)
    prices['EQUITY'] = prices['EQUITY'] * (1 + np.random.normal(0, 0.01, 60))
    
    vser = compute_forward_returns(prices, horizon_months=12, macro_data=macro_data)
    
    assert not vser.dropna().empty
    print("VSER calculation completed successfully.")
    print("Sample VSER values:\n", vser.head(12))

    # Verify reconstruction logic (Nominal = Z * sigma + Rf)
    # We can check if y_vser * sigma + Rf matches y_nominal
    y_vser = vser
    y_nominal = compute_forward_returns(prices, horizon_months=12, vol_scale=False, excess_return=False)
    
    # Get sigma and rf
    sigma = prices.pct_change().rolling(12).std() * np.sqrt(12)
    rf = macro_data['FEDFUNDS'] / 100.0
    
    reconstructed = (y_vser * sigma.values) + rf.values.reshape(-1, 1)
    
    # Check if reconstructed matches y_nominal for the indices where VSER is valid
    valid_idx = y_vser.dropna().index
    for idx in valid_idx:
        np.testing.assert_almost_equal(reconstructed.loc[idx, 'EQUITY'], y_nominal.loc[idx, 'EQUITY'], decimal=5)
    
    print("VSER Reconstruction verified successfully.")

if __name__ == "__main__":
    test_vser_calculation()
