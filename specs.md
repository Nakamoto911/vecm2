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
| **US Equities** | **Random Forest Regressor** | **Non-Linear Ensemble.** Captures complex, regime-dependent interactions (e.g., asymmetric market reactions to rate hikes) that linear models often miss. Configured with depth constraints to prevent overfitting. |
| **US Bonds** | **ElasticNetCV** | **Regularized Linear.** Bond yields typically exhibit linear relationships with fundamental drivers (Growth, Inflation) but require L1/L2 regularization to handle collinear macroeconomic inputs. |
| **Gold** | **Simple OLS** | **Parsimonious Linear.** mode Gold is modeled using a transparent linear relationship with real rates and currency drivers to ensure robustness. |

---

## 3. Feature Engineering & State Space

The model ingests raw macroeconomic data and transforms it into a "Current State" matrix ($X_t$) using stationary statistical transformations:

**Transformation Pipeline:**
1.  **Trend:** 12-month and 60-month moving averages to capture medium and long-term cycles.
2.  **Cyclical Position (Z-Score):** Deviation from the 5-year mean, normalized by 5-year volatility.
    * *Formula:* $Z_t = \frac{X_t - \mu_{60}}{\sigma_{60}}$
3.  **Historical Rank:** Rolling 10-year percentile rank (0 to 1) to contextualize current values against historical extremes.
4.  **Momentum (Slope):** Normalized rate of change over 12 and 60 months.

**Data Hygiene:**
* **Winsorization:** All input features are capped at **±3.0 Standard Deviations** to mitigate the impact of outliers and structural breaks (e.g., pandemic-era data distortions).

---

## 4. Stability Selection (Feature Selection)

To ensure trading signals are based on persistent economic drivers rather than noise, the system employs a **Stability Selection** phase prior to final estimation:

1.  **Bootstrapping:** The algorithm executes multiple iterations (e.g., 50+) of regularized regression on random subsets of the data.
2.  **Persistence Scoring:** It calculates a probability score for each variable based on how frequently it is selected with a non-zero coefficient.
3.  **Filtering:** Only features exceeding a defined **Persistence Threshold** (default 0.6) are retained for the final prediction model.

---

## 5. Risk Management: Regime Overlay

A "Nowcasting" module monitors high-frequency financial stress to adjust portfolio beta dynamically.

### A. Stress Inputs
1.  **Credit Stress:** BAA-AAA Corporate Bond Spread (Normalized Z-Score).
2.  **Yield Curve Stress:** 10Y Treasury minus Fed Funds Rate (Inverted Z-Score).

### B. Regime Logic
* **Composite Stress Score:** $0.5 \cdot Z_{Credit} + 0.5 \cdot Z_{Curve}$
* **Operational States:**
    * **CALM:** Stress Score < 1.2 (Full Target Exposure)
    * **WARNING:** 1.2 < Stress Score < 2.0 (Defensive Tilt)
    * **ALERT:** Stress Score > 2.0 (Significant Risk Reduction)

---

## 6. Strategic Allocation Logic

Portfolio weights are derived dynamically based on the interplay between expected returns and the market regime.

* **Base Allocation:** Equity 60% / Bonds 30% / Gold 10%.
* **Return Tilt:** Weights are adjusted upward or downward based on the magnitude of the predicted excess return relative to the Risk-Free Rate.
* **Regime Multiplier:** In "Alert" regimes, Equity exposure is forcibly reduced (e.g., -50%), with capital rotated into defensive assets (Bonds/Gold).
* **Hard Constraints:**
    * Equity: 20% - 80%
    * Bonds: 20% - 50%
    * Gold: 5% - 25%

---

## 7. Data Universe

* **Macroeconomic Data:** FRED-MD Database (approx. 128 monthly series covering Output, Labor, Housing, Money, and Prices).
* **Asset Pricing:**
    * **Equities:** S&P 500 Index.
    * **Bonds:** Synthetic Total Return Index derived from **GS10** (10-Year Constant Maturity Yield) assuming a constant duration ($D=7.5$).
    * **Gold:** Composite history splicing Producer Price Index (PPI) for Precious Metals with ETF data.

---

## 8. Validation Protocol

The strategy is validated using a **Purged Walk-Forward Backtest** to simulate realistic historical performance:

* **Training Window:** Expanding window with a minimum of 240 months (20 years).
* **Forecasting Horizon:** 12 Months.
* **Leakage Prevention (Purging):** The training set explicitly excludes the 12 months immediately preceding the test date to ensure no future data is used in model training.
* **Performance Metrics:** Out-of-Sample (OOS) RMSE and Information Coefficient (IC).