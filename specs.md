# Strategic Memo: Macro-Driven Strategic Asset Allocation System

## Investment Horizon: 12 Months (Rolling)

**Target Assets:** US Equities (S&P 500) / US Bonds (10Y Treasury) / Gold
**Methodology:** Direct Forward Return Prediction via Asset-Specific Machine Learning & Stability Selection

---

## 1. Objective of the Approach

The primary objective is to generate superior risk-adjusted returns by predicting **annualized 12-month forward returns** for major asset classes. The system utilizes a systematic, data-driven framework to map current macroeconomic states directly to future asset performance ($X_t \rightarrow Y_{t+12}$), prioritizing statistical stability and capital preservation over complexity.

---

## 2. Asset-Specific Model Architectures

The system employs "Best-in-Class" modeling architectures tailored to the specific statistical properties and economic drivers of each asset class:

| Asset Class | Model Architecture | Rationale |
|---|---|---|
| **US Equities** | **Random Forest Regressor** | **Non-Linear Ensemble.** Captures complex, regime-dependent interactions (e.g., asymmetric reactions to rate hikes). Configured with `max_depth=3` to ensure generalization. |
| **US Bonds** | **ElasticNetCV** | **Regularized Linear.** Handles collinear macroeconomic inputs using L1/L2 regularization to identify robust fundamental drivers. |
| **Gold** | **Linear Regression (OLS)** | **Parsimonious Linear.** Gold is modeled using a transparent linear relationship with real rates and currency drivers to ensure robustness. |

---

## 3. Feature Engineering & State Space

The model ingests raw macroeconomic data and transforms it into a "Current State" matrix ($X_t$) using stationary statistical transformations:

**Transformation Pipeline:**
1.  **Trend:** 12-month and 60-month moving averages for medium and long-term cycles.
2.  **Cyclical Position (Z-Score):** Deviation from the 5-year mean, normalized by 5-year volatility.
    * *Formula:* $Z_t = \frac{X_t - \mu_{60}}{\sigma_{60}}$
3.  **Historical Rank:** Rolling 10-year percentile rank (0 to 1) to contextualize current values.
4.  **Momentum (Slope):** Normalized rate of change over 12 and 60 months.

**Data Hygiene:**
* **Winsorization:** All input features are capped at **±3.0 Standard Deviations** (fit on training data) to mitigate outliers.

---

## 4. Stability Selection (Feature Selection)

To ensure trading signals are based on persistent economic drivers rather than noise, the system employs a **Stability Selection** phase prior to final estimation:

1.  **Bootstrapping:** Executes 50 iterations (30 in backtests) of ElasticNet regression ($l1\_ratio=0.5$) on random 70% subsets of the data.
2.  **Persistence Scoring:** Calculates the frequency (%) each variable is selected with a non-zero coefficient.
3.  **Filtering:** Only features exceeding a **Persistence Threshold** (default 0.6) are retained.

---

## 5. Risk Management: Regime Overlay

A "Nowcasting" module monitors stress to adjust portfolio beta dynamically.

### A. Stress Inputs (Rolling 60-Month Z-Scores)
1.  **Credit Stress:** BAA-AAA Corporate Bond Spread.
2.  **Yield Curve Stress:** 10Y Treasury minus Fed Funds Rate (Inverted).

### B. Regime Logic
* **Composite Stress Score:** $0.5 \cdot Z_{Credit} + 0.5 \cdot Z_{Curve}$
* **Operational States:**
    * **CALM:** Stress Score < 1.2 (Full Target Exposure)
    * **WARNING:** 1.2 < Stress Score < 2.0 (Defensive Tilt: -25% Equity Beta)
    * **ALERT:** Stress Score > 2.0 (High Defensive: -50% Equity Beta)

---

## 6. Strategic Allocation Logic

Portfolio weights are derived dynamically based on expected returns and market regime.

* **Base Weights:** Equity 60% / Bonds 30% / Gold 10%.
* **Return Tilt:** Weights are adjusted based on predicted excess returns: 
  $Target = Base \cdot RegimeMult \cdot (1 + (PredictedReturn - RF) \cdot 5)$
* **Hard Constraints:**
    * Equity: 20% - 80% | Bonds: 20% - 50% | Gold: 5% - 25%
* **Normalization:** Final weights are scaled to sum to 100%.

---

## 7. Data Universe

* **Macroeconomic Data:** FRED-MD Database (approx. 128 monthly series).
* **Asset Pricing:**
    * **Equities:** S&P 500 Index.
    * **Bonds:** Synthetic Total Return Index from GS10 assumes constant duration ($D=7.5$).
    * **Gold:** PPI for Gold (WPU1022) as long-term proxy.

---

## 8. Validation Protocol

The strategy is validated using a **Purged Walk-Forward Backtest**:

* **Training Window:** Expanding window with a minimum of 240 months.
* **Leakage Prevention (Purging):** Excludes 12 months immediately preceding the test date.
* **Overlap Adjustment (HAC):** 
    * Forecast errors are adjusted for 12-month overlap by scaling standard errors by $\sqrt{12}$.
    * **Effective Degrees of Freedom:** Adjusted for autocorrelation ($N_{eff} = N / 12$).
* **Performance Metrics:** OOS Information Coefficient (IC) and RMSE.