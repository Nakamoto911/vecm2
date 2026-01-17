import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings

from opus import load_fred_md_data, load_asset_data, prepare_macro_features, compute_forward_returns

# Suppress convergence warnings for cleaner output during benchmark
warnings.filterwarnings("ignore")

# ==========================================
# 1. Preprocessing & Transformers
# ==========================================

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Caps features at a specific Z-score threshold.
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.means_ = X_df.mean()
        self.stds_ = X_df.std()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            z_score = (X_df[col] - self.means_[col]) / (self.stds_[col] + 1e-9)
            X_df[col] = X_df[col].mask(z_score > self.threshold, self.means_[col] + self.threshold * self.stds_[col])
            X_df[col] = X_df[col].mask(z_score < -self.threshold, self.means_[col] - self.threshold * self.stds_[col])
        return X_df.values

def select_features_elastic_net(y: pd.Series, X: pd.DataFrame, 
                                 n_iterations: int = 30,
                                 sample_fraction: float = 0.7,
                                 threshold: float = 0.6,
                                 l1_ratio: float = 0.5) -> tuple:
    """
    Implement Stability Selection via bootstrapping.
    """
    from sklearn.linear_model import ElasticNet
    
    # Pre-clean: Drop columns that are all NaN or have any NaN in the provided sample
    X = X.dropna(axis=1)
    
    if X.empty:
        return [], pd.Series(dtype=float)

    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    n_features = X.shape[1]
    selection_counts = pd.Series(0, index=X.columns)
    
    # 1. Bootstrapping Loop
    for _ in range(n_iterations):
        # Subsample indices
        sample_size = int(len(y) * sample_fraction)
        indices = np.random.choice(len(y), size=sample_size, replace=False)
        
        y_sample = y.iloc[indices]
        X_sample = X_scaled.iloc[indices]
        
        # Fit a simple ElasticNet (faster than CV for each bootstrap)
        # We use a small alpha for selection
        model = ElasticNet(alpha=0.01, l1_ratio=l1_ratio, max_iter=5000)
        model.fit(X_sample, y_sample)
        
        # Record non-zero coefficients
        selection_counts[model.coef_ != 0] += 1
        
    # 2. Calculate Probabilities
    selection_probs = selection_counts / n_iterations
    
    # 3. Apply Threshold
    selected = selection_probs[selection_probs > threshold].index.tolist()
    
    # 4. Fallback Logic: ensure at least one feature
    if not selected and not selection_probs.empty:
        selected = [selection_probs.idxmax()]
    
    return selected, selection_probs

# ==========================================
# 2. Custom Model: Regime Switching
# ==========================================

class RegimeSwitchingModel(BaseEstimator, RegressorMixin):
    """
    Switches between two models based on a regime indicator (e.g., Inflation Moving Average).
    """
    def __init__(self, model_low, model_high, regime_col_idx=0, threshold=0.04):
        self.model_low = model_low
        self.model_high = model_high
        self.regime_col_idx = regime_col_idx
        self.threshold = threshold
        self.trained_low = None
        self.trained_high = None

    def fit(self, X, y):
        regime_vals = X[:, self.regime_col_idx]
        
        mask_high = regime_vals > self.threshold
        mask_low = ~mask_high
        
        self.trained_low = clone(self.model_low)
        self.trained_high = clone(self.model_high)
        
        if mask_low.any():
            self.trained_low.fit(X[mask_low], y[mask_low])
        if mask_high.any():
            self.trained_high.fit(X[mask_high], y[mask_high])
            
        return self

    def predict(self, X):
        regime_vals = X[:, self.regime_col_idx]
        preds = np.zeros(len(X))
        
        mask_high = regime_vals > self.threshold
        mask_low = ~mask_high
        
        if mask_low.any():
            preds[mask_low] = self.trained_low.predict(X[mask_low])
        if mask_high.any():
            preds[mask_high] = self.trained_high.predict(X[mask_high])
            
        return preds

# ==========================================
# 3. Validation Logic (Purged Walk-Forward)
# ==========================================

def run_benchmarking_engine(X, y, start_idx, step=12, horizon=12, regime_idx=0):
    """
    Strict Walk-Forward Validation with Purging.
    """
    
    models = {
        "Baseline (Mean)": DummyRegressor(strategy='mean'),
        "Simple OLS": LinearRegression(),
        "Robust (Huber)": HuberRegressor(),
        "Regularized (ENet)": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5),
        "Non-Linear (RF)": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "Regime-Based (Inf)": RegimeSwitchingModel(
            model_low=LinearRegression(), 
            model_high=LinearRegression(),
            regime_col_idx=regime_idx, 
            threshold=0.04 
        )
    }
    
    results = {name: [] for name in models.keys()}
    actuals = []
    
    # print(f"Starting Walk-Forward Benchmark (N={len(X)}, Start={start_idx}, Step={step}, Purge={horizon})")
    
    for i in range(start_idx, len(X) - 1, step):
        # 1. Define Training Set
        train_X_raw = X.iloc[:i]
        train_y = y.iloc[:i]
        
        # 2. Purge step
        purge_idx = i - horizon
        if purge_idx <= 0:
            continue
            
        train_X_purged = train_X_raw.iloc[:purge_idx]
        train_y_purged = train_y.iloc[:purge_idx]
        
        test_X_raw = X.iloc[i : min(i + step, len(X))]
        test_y = y.iloc[i : min(i + step, len(X))]
        
        if test_y.empty:
            break
            
        actual_vals = test_y.values.flatten()
        actuals.extend(actual_vals)
        
        # 3. [PIT] Dynamic Feature Selection
        # Use ONLY train_X_purged and train_y_purged to avoid look-ahead bias
        stable_feats, _ = select_features_elastic_net(
            train_y_purged, train_X_purged,
            threshold=0.6,
            l1_ratio=0.5
        )
        if not stable_feats:
            stable_feats = train_X_purged.columns.tolist()
            
        train_X_sel = train_X_purged[stable_feats].dropna(axis=1) # Second pass drop just in case
        test_X_sel = test_X_raw[stable_feats]
        
        # 4. Apply Winsorization (Fit on purged training)
        win = Winsorizer(threshold=3.0)
        train_X = win.fit_transform(train_X_sel)
        test_X = win.transform(test_X_sel)
        
        # 4. Fit & Predict each model
        for name, model in models.items():
            m = clone(model)
            # Handle RegimeSwitchingModel differently if it relies on a specific column index
            if isinstance(m, RegimeSwitchingModel):
                # For regime switching, we might need the regime column (Inflation) to be present
                # Ideally, we force 'CPIAUCSL_MA60' to be in stable_feats if using this model.
                # For simplicity here, we skip RS if regime col missing, or pass it separately.
                # Assuming generic regression for now as RS model handles full X internally usually?
                # Actually RS model splits data based on col idx. If features change, idx changes.
                # Disabling RS model for dynamic selection benchmark or running it on FULL features (hybrid).
                # To imply "best practice", we skip it or use fallback.
                # Let's fallback to fitting it on clean data if dimensions match (risky).
                # We simply skip RS for this strict PIT test to avoid indexing errors.
                continue
                
            m.fit(train_X, train_y_purged.values.ravel())
            
            preds = m.predict(test_X)
            
            # Safety Rail: Output Clipping (±30%)
            preds = np.clip(preds, -0.30, 0.30)
            
            results[name].extend(preds)
            
    # Calculate Metrics
    summary = []
    for name, preds in results.items():
        preds = np.array(preds)
        valid_actuals = np.array(actuals[:len(preds)])
        
        if len(preds) == 0:
            continue
            
        mse = mean_squared_error(valid_actuals, preds)
        rmse = np.sqrt(mse)
        ic, _ = spearmanr(valid_actuals, preds)
        
        summary.append({
            "Model": name,
            "OOS IC (Corr)": ic,
            "OOS RMSE": rmse
        })
        
    df_summary = pd.DataFrame(summary).sort_values("OOS IC (Corr)", ascending=False)
    return df_summary, models

# ==========================================
# 4. Output Formatter
# ==========================================

def display_current_regime_model(name, model, X, y, feature_names, asset_name):
    """
    Fits model on the FULL dataset and displays its parameters/importance.
    """
    print(f"\n>>> FINAL MODEL DETAILS: {asset_name} ({name})")
    
    # Pre-process full data
    win = Winsorizer(threshold=3.0)
    X_clean = win.fit_transform(X)
    
    # Fit
    m = clone(model)
    m.fit(X_clean, y.values.ravel())
    
    if isinstance(m, (LinearRegression, HuberRegressor, ElasticNetCV)):
        intercept = m.intercept_
        coefs = m.coef_
        
        equation = f"Return = {intercept:.4f}"
        # Only show top 10 features for readability
        sorted_indices = np.argsort(np.abs(coefs))[::-1]
        for idx in sorted_indices[:10]:
            if np.abs(coefs[idx]) > 0.0001:
                equation += f" + ({coefs[idx]:.4f} * {feature_names[idx]})"
        print(f"EQUATION:\n{equation}")
        
    elif isinstance(m, RandomForestRegressor):
        importances = m.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("TOP 10 VARIABLES (FEATURE IMPORTANCE):")
        for idx in sorted_indices[:10]:
            print(f"- {feature_names[idx]}: {importances[idx]:.4f}")
            
    elif isinstance(m, RegimeSwitchingModel):
        print(f"REGIME SWITCHING ANALYSIS (Threshold: {m.threshold})")
        # Details for low regime
        if m.trained_low and hasattr(m.trained_low, 'coef_'):
            print(f"  [NORMAL REGIME] Intercept: {m.trained_low.intercept_:.4f}")
            sorted_idx = np.argsort(np.abs(m.trained_low.coef_))[::-1]
            for idx in sorted_idx[:5]:
                print(f"    - {feature_names[idx]}: {m.trained_low.coef_[idx]:.4f}")
        
        # Details for high regime
        if m.trained_high and hasattr(m.trained_high, 'coef_'):
            print(f"  [HIGH REGIME] Intercept: {m.trained_high.intercept_:.4f}")
            sorted_idx = np.argsort(np.abs(m.trained_high.coef_))[::-1]
            for idx in sorted_idx[:5]:
                print(f"    - {feature_names[idx]}: {m.trained_high.coef_[idx]:.4f}")
    
    elif isinstance(m, DummyRegressor):
        print(f"EQUATION: Return = {y.mean():.4f} (Historical Mean)")

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # Load Real Data
    print("Loading Real Data from opus.py pipeline...")
    macro_data = load_fred_md_data()
    asset_prices = load_asset_data()
    
    if macro_data.empty or asset_prices.empty:
        print("Error: Could not load real data.")
    else:
        # Align data
        common_idx = macro_data.index.intersection(asset_prices.index)
        macro_data = macro_data.loc[common_idx]
        asset_prices = asset_prices.loc[common_idx]
        
        # Prepare Features and Targets (12-month horizon)
        horizon = 12
        macro_features = prepare_macro_features(macro_data)
        y_forward = compute_forward_returns(asset_prices, horizon_months=horizon)
        
        # Align again
        valid_idx = macro_features.index.intersection(y_forward.index)
        X_full = macro_features.loc[valid_idx]
        y_full = y_forward.loc[valid_idx]
        
        # Identify inflation index for regime switching (CPIAUCSL_MA60)
        inf_col = "CPIAUCSL_MA60" if "CPIAUCSL_MA60" in X_full.columns else X_full.columns[0]
        regime_idx = X_full.columns.get_loc(inf_col)
        
        for asset in ['EQUITY', 'BONDS', 'GOLD']:
            print(f"\n" + "="*60)
            print(f" ASSET CLASS: {asset}")
            print("="*60)
            
            y_asset = y_full[asset].dropna()
            X_asset = X_full.loc[y_asset.index]
            
            # Start benchmark after 20 years of data
            benchmark_table, model_templates = run_benchmarking_engine(
                X_asset, y_asset, 
                start_idx=240, 
                step=12, 
                horizon=horizon,
                regime_idx=regime_idx
            )
            
            print("\nOOS PERFORMANCE LEADERBOARD:")
            print(benchmark_table.to_string(index=False))
            
            if not benchmark_table.empty:
                winner_name = benchmark_table.iloc[0]["Model"]
                winner_model = model_templates[winner_name]
                
                display_current_regime_model(
                    winner_name, 
                    winner_model, 
                    X_asset, 
                    y_asset, 
                    X_asset.columns, 
                    asset
                )
            else:
                print(f"No results for {asset}")
