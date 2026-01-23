import pandas as pd
import numpy as np
from prediction_metrics import generate_llm_report

def test_report_generation():
    # Mock data
    assets = ['EQUITY', 'BONDS', 'GOLD']
    y_live = pd.DataFrame(np.random.randn(100, 3), columns=assets)
    y_live.index = pd.date_range('2010-01-01', periods=100, freq='M')
    
    prediction_results = {}
    for asset in assets:
        # Create mock OOS results
        dates = y_live.index[50:]
        n = len(dates)
        preds = y_live[asset].loc[dates] + np.random.normal(0, 0.1, n)
        oos = pd.DataFrame({
            'predicted_return': preds,
            'lower_ci': preds - 0.05,
            'upper_ci': preds + 0.05
        }, index=dates)
        prediction_results[asset] = oos
        
    # Mock model stats
    model_stats = {
        'EQUITY': {'model': 'XGB', 'importance': pd.Series({'feature_A': 0.5, 'feature_B': 0.3})},
        'BONDS': {'model': 'ElasticNet', 'coefficients': pd.Series({'feature_C': 0.1, 'const': 0.01}), 'intercept': 0.01},
        'GOLD': {'model': 'OLS', 'coefficients': pd.Series({'feature_D': -0.2}), 'intercept': -0.01}
    }
        
    report = generate_llm_report(prediction_results, y_live, 0.90, model_stats)
    print("Generated Report Preview:")
    print("-" * 40)
    print(report[:2000] + "...")
    print("-" * 40)
    
    assert "# MODEL OUT-OF-SAMPLE PERFORMANCE REPORT" in report
    assert "### Model Specification" in report
    assert "Architecture: XGBoost" in report
    assert "Architecture: ElasticNet" in report
    
    print("Test passed!")

if __name__ == "__main__":
    test_report_generation()
