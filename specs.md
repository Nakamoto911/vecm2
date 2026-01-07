# Strategic Memo: High-Dimensional Adaptive Sparse Elastic Net VECM

## Investment Horizon: 5 to 10 years

**Target Assets:** US Equities / Bonds / Gold

**Methodology:** Adaptive Sparse VECM with Elastic Net, Kernel Dictionary, and High-Frequency Nowcasting Overlay

---

## 1. Objective of the Approach

The primary objective of this approach is capital preservation while targeting systematically superior returns to the risk-free rate. The strategy is calibrated to generate target returns between 5% and 10% annually, with rigorous management constraints:

**Light Management and Minimal Rotation:** The strategy favors a dynamic "buy-and-hold" approach. Portfolio rebalancing occurs on a quarterly basis to limit transaction costs and tax impact.

**Flexibility in Crisis Situations:** As an exception to the quarterly rule, the model can trigger immediate rotation if the "Nowcasting" module detects a structural break or imminent systemic risk.

**Minimal Volatility and Limited Drawdowns:** Optimization of allocation to smooth the return curve and early identification of reversal risks to minimize maximum losses.

The success of this management relies on identifying the economic cycle, defined by the interaction between long-term movements (levels) and short-term movements (variations).

---

## 2. Architecture of the Temporal Kernel Dictionary

To capture cycles without rigidity, the model uses Temporal Kernels:

**Lags 1 to 6:** Dense structure for immediate reactivity (essential for crisis detection).

**Anchors 12, 24, 36, 48, 60:** Gaussian-weighted averages to capture cyclical shock waves without information "black holes."

---

## 3. Complementary Module: High-Frequency Nowcasting Overlay

This module acts as a tactical emergency brake. It monitors daily market stress to cut risk exposure before VECM macroeconomic data confirms the crisis.

### A. Stress Indicators (HF Inputs)

**Credit (HY-IG Spread):** Rate spread between risky and solid companies. A sharp widening warns of increased default risk.

**Fear (VIX Structure):** VIX / VIX3M ratio. A shift into Backwardation (VIX > VIX3M) signals immediate panic.

**Liquidity (DXY & 10-Year Rates):** Simultaneous increases in the Dollar and real rates signal global liquidity tightening.

### B. Composite Stress Index (CSI)

Variables are normalized into Z-Scores ($Z_t$) over 12 months.

$$CSI_t = 0.4 \cdot Z_{Credit} + 0.4 \cdot Z_{Vol} + 0.2 \cdot Z_{Liquidity}$$

### C. Decision Matrix (Overlay)

| VECM Signal (Macro) | Nowcast Signal (HF Stress) | Strategic Action |
|---|---|---|
| Bullish (Bull) | Calm | 100% Target Exposure. |
| Bullish (Bull) | High Stress | Tactical Hedge: Reduce exposure by 50%. |
| Bearish (Bear) | Calm | Progressive Exit: Sell on bounces. |
| Bearish (Bear) | High Stress | Full Defense: 100% Cash / Gold / Short-term Bonds. |

---

## 4. Algorithm Steps

**Step 1: Ingestion and Preprocessing**

Load FRED-MD (monthly) and market data (daily). Apply log-level and difference transformations.

Output: Cleaned dataset, separated into levels ($y_t$) and variations ($\Delta y_t$).

**Step 2: Daily Nowcasting Check**

Calculate the Composite Stress Index (CSI) daily.

Output: "Sentinel" status (Alert or Calm) dictating maintenance or reduction of exposure.

**Step 3: Cointegration Rank Identification**

Weighted Johansen test on level variables to identify long-term equilibrium.

Output: Rank $r$ and cointegration vectors $\beta$ (definition of equilibrium value).

**Step 4: Estimation via Adaptive Elastic Net**

Generate final predictive equations by applying dual penalty (L1/L2) on variations filtered by kernels.

Output: Stable equations and Heatmap of coefficients $\Gamma$ (isolates current active drivers).

**Step 5: Extraction of Error Correction Term (ECT)**

Calculate $ECT_t = \beta' y_{t-1}$. Measure of deviation from macroeconomic fair value.

Output: Imbalance score (Error Term) indicating the strength of pull back to equilibrium.

**Step 6: Signal Generation and Rebalancing**

Quarterly synthesis (or immediate in case of Nowcast alert) to adjust allocation.

Output: List of orders and new target portfolio weights.

---

## 5. Stability and Storytelling (Group Effect)

The use of Elastic Net ensures that the economic "narrative" remains coherent:

**Temporal Coherence:** Avoids jumping from one variable to another month to month, reducing unnecessary turnover.

**Grouped Variables:** Signal confirmation through coherent blocks (e.g., the "Labor Market" block is selected in its entirety).

---

## 6. Cycle Signatures (Example of Algorithm Output)

| Block / Module | Grouped Variables Selected | Status / Regime | Strategic Interpretation |
|---|---|---|---|
| Nowcast (HF) | HY-IG Spread & VIX | Calm | No immediate liquidity stress. |
| Labor (Macro) | PAYEMS & USPRIV (Employment) | Plateauing | Late-cycle expansion detected. |
| Output (Macro) | INDPRO & IPFINAL (Production) | Slowdown | Convergence of production indicators. |
| Prices (Macro) | CPI & PPI (Inflation) | "Sticky" | Persistent inflationary pressures. |

---

## 7. Validation Protocol and Resilience

**Hyperparameter Optimization:** Choice of L1/L2 ratio to favor selection stability (Ridge-heavy) over extreme sparsity.

**Cointegration Monitoring:** In case of major $\beta$ instability (structural break), the model switches to "Preservation" mode (Gold/Cash) until the next stabilization cycle.