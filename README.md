# SMA Crossover Backtester

A vectorised Python backtesting engine for a Simple Moving Average (SMA) crossover strategy, tested across 10 ASX blue-chip equities over 5 years of historical data.

Built to explore quantitative trading concepts: signal generation, risk-adjusted performance measurement, statistical significance testing, and parameter sensitivity analysis.

---

## What It Does

1. **Downloads** 5 years of real historical price data for 10 ASX stocks via Yahoo Finance
2. **Generates signals** — goes long (+1) when the fast SMA crosses above the slow SMA, short (-1) otherwise
3. **Computes performance metrics** — Sharpe ratio, max drawdown, annualised return, win rate
4. **Tests statistical significance** — t-test to distinguish genuine edge from noise
5. **Runs a parameter sweep** — tests 9 fast/slow window combinations to check strategy robustness

---

## Strategy: SMA Crossover

```
Fast SMA(20) > Slow SMA(50)  →  LONG  (uptrend)
Fast SMA(20) < Slow SMA(50)  →  SHORT (downtrend)
```

Signals are **shifted forward by 1 day** before computing returns. This prevents look-ahead bias — in practice, you observe today's close and trade at tomorrow's open.

---

## Metrics Explained

| Metric | What It Means |
|---|---|
| **Sharpe Ratio** | Risk-adjusted return: `(mean return / std dev) × √252`. Annualised. >1.0 acceptable, >2.0 strong |
| **Max Drawdown** | Largest peak-to-trough loss. A strategy with -60% drawdown is unusable regardless of total return |
| **Win Rate** | % of trading days with positive return |
| **Annualised Return** | Compound annual growth rate over the test period |
| **P-Value** | From a t-test vs. zero mean. P < 0.05 means returns are unlikely to be random noise |
| **Statistically Significant** | Whether the strategy's edge clears the 95% confidence threshold |

---

## Outputs

- **Console table** — full metrics for all 10 tickers
- **`cumulative_returns.png`** — $1 growth chart for all tickers over the test period
- **`parameter_sweep_CBA_AX.png`** — 3×3 grid showing Sharpe ratio across 9 parameter combinations, testing strategy robustness vs. overfitting

---

## Running It

```bash
pip install -r requirements.txt
python main.py
```

---

## Project Structure

```
backtester/
├── strategy.py       # SMACrossoverStrategy class — signal generation logic
├── backtester.py     # Backtester class — data download, execution, metrics
├── main.py           # Entry point — runs sweep, prints table, saves charts
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Why shift signals by 1 day?**
Without the shift, you'd be trading on the same candle that generated the signal — physically impossible in live trading. Shifting by 1 simulates realistic execution.

**Why test statistical significance?**
A strategy can show positive historical returns purely by chance, especially over short periods. A t-test against zero mean checks whether the observed returns are statistically distinguishable from noise. If p > 0.05, the strategy has no provable edge.

**Why a parameter sweep?**
A strategy that only works with exactly `fast=20, slow=50` is likely overfit to historical data. A robust strategy should show consistent performance across a range of parameters. The sweep visualises this.

**Why multiple tickers?**
Testing on one stock risks finding a strategy that worked by coincidence for that specific instrument. Cross-sectional testing across 10 stocks gives a more honest signal about strategy viability.

---

## Limitations (Honest Assessment)

- **No transaction costs or slippage** — real trading incurs brokerage fees and market impact. These eat significantly into returns, especially for short-term strategies.
- **Survivorship bias** — using current ASX constituents excludes companies that were delisted. This overstates historical performance.
- **SMA crossover is a known strategy** — its alpha has likely been arbitraged away. The point of this project is to learn backtesting infrastructure, not to discover a profitable strategy.
- **No position sizing** — assumes 100% portfolio allocation per signal. Real strategies use Kelly criterion or volatility-targeting for sizing.

---

## What I'd Add Next

- Transaction cost model (fixed + percentage brokerage per trade)
- Benchmark comparison (strategy vs buy-and-hold vs ASX 200 index)
- Walk-forward validation to test out-of-sample performance
- Additional strategies (RSI mean reversion, Bollinger Bands)
- Kelly criterion position sizing

---

## Requirements

Python 3.8+ | No paid data subscriptions required.
