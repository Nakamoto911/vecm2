# Strategic Memo: Macro-Driven Strategic Asset Allocation System

## 1. Objective of the Approach

The primary objective is to generate superior risk-adjusted returns by predicting **annualized 12-month forward returns** for major asset classes. The system utilizes a systematic, data-driven framework to map current macroeconomic states directly to future asset performance ($X_t \rightarrow Y_{t+12}$), prioritizing statistical stability and capital preservation over complexity.

---

## 2. Data Universe & PIT Architecture

The system employs a dual-path data architecture to ensure backtest integrity while providing the most accurate live predictions.

### 2.1 Dual-Path Architecture
1.  **Backtest Engine**: Exclusively uses `PIT_Macro_Features.csv`. This ensures the model only "sees" data as it would have appeared to a researcher at any historical point $t$.
2.  **Live Prediction Terminal**: Uses the latest available FRED-MD snapshot (e.g., `2025-11-MD.csv`). This provides the most refined recent data for the current month's signal.

### 2.2 PIT Vintage Construction (Diagonalization)
The historical vintage matrix is built via a "Diagonalization" process implemented in [pit_builder.py](pit_builder.py):
1.  **Sourcing**: Extracts historical monthly CSV vintages (dating back to the 1960s) from the St. Louis Fed research database.
2.  **Nowcast Extraction**: For every month $v$, the builder loads the specific vintage file released during that month and extracts the **last valid row**.
3.  **Conservative Lag Protocol**: To prevent inadvertent "peeking" at mid-month data releases, the system enforces a **1-month hard lag**. A vintage labeled "November 2025" becomes active in the simulation only on **December 1st, 2025**.

### 2.3 Assets & Data Sources
*   **Macroeconomic Data**: FRED-MD Database (approx. 128 monthly series).
*   **Equities**: S&P 500 Index (via Yahoo Finance).
*   **Bonds**: Synthetic Total Return Index from GS10 assuming constant duration ($D=7.5$).
*   **Gold**: PPI for Gold (WPU1022) as long-term proxy, spliced with GLD pricing.

---

## 3. Data Pipeline

The pipeline ensures strict stationarity and isolates unique macro signals from common drivers.

### 3.1 Transformation (Stationarity-First)
All RAW FRED-MD variables are transformed using McCracken & Ng (2016) codes:
1. **Level** | 2. **$\Delta$** | 3. **$\Delta^2$** | 4. **$\ln$** | 5. **$\Delta \ln$** | 6. **$\Delta^2 \ln$** | 7. **$\Delta$ % Change**

### 3.2 Factor Stripping (Orthogonalization)
Transformed series are orthogonalized against the "Big 4" common drivers to extract residual signals ($r_t$):
- **Inflation** (CPIAUCSL) | **Growth** (INDPRO) | **Liquidity** (M2SL) | **Policy** (FEDFUNDS)
- *Formula*: $x_{transformed} = \beta_{big4} \cdot \text{Big4} + r_t$

### 3.3 Feature Expansion
Systematic expansion of residuals and levels:
- **Momentum**: 12-month rolling slope.
- **Acceleration (Impulse)**: Change in momentum ($\Delta$ Slope).
- **Volatility**: 60-month rolling standard deviation.
- **Lags**: 1-month and 3-month lookbacks.
- **Symbolic Ratios**: Golden ratios (e.g., M2/Growth) to capture structural shifts.

---

## 4. Modeling & Estimation

### 4.1 Stability Selection
To mitigate overfitting, the system executes **Stability Selection** (Randomized Lasso) on 50 bootstrapped subsamples. Only features with a **Persistence Score > 70%** are retained.

### 4.2 Modeling Architectures
The system leverages architectures optimized for the signal-to-noise ratio of each asset class:
- **Equities**: **XGBoost** (Gradient Boosting) with shallow depth (3).
- **Bonds**: **ElasticNet** (Regularized Linear) for sparse feature selection.
- **Gold**: **LSTM** (Recurrent Neural Network) or OLS, focusing on sequence dependencies.

### 4.3 Uncertainty Quantification
Confidence intervals are generated using **HAC-Adjusted Standard Errors**. Because targets are 12-month overlapping returns, the standard error is inflated by $\sqrt{12}$ and effective degrees of freedom are reduced ($N_{eff} = N/12$).

---

## 5. Portfolio Construction & Risk Management

### 5.1 Regime Overlay (Nowcasting)
A composite stress score monitors **Credit Spreads** (BAA-AAA) and **Yield Curve** (10Y-FFR):
*   **CALM**: Stress < 1.2 (Full Beta)
*   **WARNING**: 1.2 < Stress < 2.0 (-25% Equity Tilt)
*   **ALERT**: Stress > 2.0 (-50% Equity Tilt)

### 5.2 Strategic Allocation
*   **Base Weights**: Equity 60% / Bonds 30% / Gold 10%.
*   **Allocation Tilt**: $Base \cdot RegimeMult \cdot (1 + (Predicted_Return - RF) \cdot 5)$
*   **Hard Constraints**: Equity [20-80%], Bonds [20-50%], Gold [5-25%].

---

## 6. Validation Protocol

Validated using a **Purged Walk-Forward Backtest**:
*   **Purging**: 12-month "purge gap" between training and testing data to eliminate look-ahead bias from overlapping returns.
*   **Data Hygiene**: **Winsorization** (caps at ±3.0 SD) fit strictly on the training set.
*   **Metrics**: recursive OOS Spearman Correlation (IC) and RMSE.