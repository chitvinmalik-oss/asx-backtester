"""
main.py
Runs the SMA crossover backtester across a portfolio of ASX stocks.
Generates a performance summary table and a cumulative returns chart.

Run:
    pip install yfinance pandas numpy scipy matplotlib
    python main.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import yfinance as yf

from backtester import Backtester
from strategy import SMACrossoverStrategy

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

# ASX blue-chip tickers (.AX suffix = ASX listed)
TICKERS = [
    "BHP.AX",   # BHP Group — mining
    "CBA.AX",   # Commonwealth Bank — financials
    "CSL.AX",   # CSL Limited — healthcare
    "WES.AX",   # Wesfarmers — retail/industrial
    "ANZ.AX",   # ANZ Bank — financials
    "WBC.AX",   # Westpac — financials
    "NAB.AX",   # NAB — financials
    "MQG.AX",   # Macquarie Group — financials
    "TLS.AX",   # Telstra — telecoms
    "RIO.AX",   # Rio Tinto — mining
]

START_DATE = "2019-01-01"
END_DATE = "2024-01-01"  # 5 years of data

# SMA parameters — fast crosses slow to generate signals
FAST_WINDOW = 20   # ~1 month
SLOW_WINDOW = 50   # ~2.5 months


# ── Run backtests ─────────────────────────────────────────────────────────────

def run_all(tickers, start, end, fast, slow):
    """
    Run the strategy across all tickers and collect results.

    Returns:
        results (list of dict): Metrics per ticker.
        all_returns (dict): Raw return series per ticker for plotting.
    """
    strategy = SMACrossoverStrategy(fast_window=fast, slow_window=slow)
    results = []
    all_returns = {}

    for ticker in tickers:
        try:
            bt = Backtester(ticker, start, end)
            returns = bt.run(strategy)
            metrics = bt.metrics(returns)
            results.append(metrics)
            all_returns[ticker] = returns
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    return results, all_returns


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_cumulative_returns(all_returns, fast, slow):
    """
    Plot cumulative strategy returns for all tickers on one chart.
    Cumulative return = compound growth of $1 invested at start.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker, returns in all_returns.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative, linewidth=1.2, label=ticker)

    # Reference line: $1 stayed flat (no edge = flat line)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8,
               label="No edge (flat)")

    ax.set_title(
        f"SMA Crossover Strategy — Cumulative Returns\n"
        f"Fast={fast}d / Slow={slow}d  |  ASX Blue Chips  |  {START_DATE} to {END_DATE}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return ($1 invested)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nChart saved to cumulative_returns.png")


def plot_parameter_sweep(ticker, start, end):
    """
    Sweep over different fast/slow window combinations for one ticker.
    Shows how sensitive the strategy is to parameter choice —
    a robust strategy should work across a range, not just one lucky setting.
    This guards against overfitting (curve-fitting to historical noise).
    """
    fast_windows = [10, 20, 30]
    slow_windows = [50, 100, 200]

    fig, axes = plt.subplots(len(fast_windows), len(slow_windows),
                             figsize=(16, 10), sharex=True, sharey=True)

    bt = Backtester(ticker, start, end)

    for i, fast in enumerate(fast_windows):
        for j, slow in enumerate(slow_windows):
            ax = axes[i][j]
            strategy = SMACrossoverStrategy(fast_window=fast, slow_window=slow)
            returns = bt.run(strategy)
            cumulative = (1 + returns).cumprod()
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)

            ax.plot(cumulative.index, cumulative, linewidth=1, color="steelblue")
            ax.axhline(1.0, color="red", linestyle="--", linewidth=0.7)
            ax.set_title(f"F={fast}/S={slow}\nSharpe={sharpe:.2f}", fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Parameter Sweep — {ticker}  |  {start} to {end}\n"
        "Robust strategy should show consistent Sharpe across parameter combinations",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"parameter_sweep_{ticker.replace('.', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Parameter sweep saved to parameter_sweep_{ticker.replace('.', '_')}.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SMA CROSSOVER BACKTESTER")
    print(f"  Strategy: Fast SMA({FAST_WINDOW}) / Slow SMA({SLOW_WINDOW})")
    print(f"  Universe: ASX Blue Chips")
    print(f"  Period:   {START_DATE} to {END_DATE}")
    print("=" * 60)
    print("\nDownloading data and running backtests...\n")

    results, all_returns = run_all(
        TICKERS, START_DATE, END_DATE, FAST_WINDOW, SLOW_WINDOW
    )

    # ── Print summary table ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))

        # Count how many strategies had statistically significant returns
        sig_count = df["Statistically Significant"].str.contains("Yes").sum()
        print(f"\n{sig_count}/{len(df)} strategies statistically significant (p < 0.05)")
        print("\nNote: statistical significance in backtesting does NOT guarantee")
        print("future performance. Always account for transaction costs,")
        print("slippage, and overfitting risk before drawing conclusions.")

    # ── Plots ──
    print("\nGenerating charts...")
    if all_returns:
        plot_cumulative_returns(all_returns, FAST_WINDOW, SLOW_WINDOW)
        # Parameter sweep for the most data-rich ticker
        plot_parameter_sweep("CBA.AX", START_DATE, END_DATE)

    print("\nDone.")


if __name__ == "__main__":
    main()
