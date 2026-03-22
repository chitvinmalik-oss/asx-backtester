"""
backtester.py
Core backtesting engine. Downloads historical price data, applies a strategy,
and computes performance metrics including statistical significance testing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


class Backtester:
    """
    Runs a trading strategy against historical price data and computes
    performance metrics.

    Design decision: signals are shifted forward by 1 day before computing
    returns. This is critical — it prevents look-ahead bias. In reality, you
    observe today's closing price and act tomorrow at the open. Without the
    shift, you'd be trading on information you couldn't have had yet.

    Args:
        ticker (str): Ticker symbol e.g. 'BHP.AX' for ASX, 'AAPL' for US.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    """

    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.prices = self._download()

    def _download(self):
        """Download adjusted closing prices from Yahoo Finance."""
        data = yf.download(self.ticker, start=self.start, end=self.end,
                           progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No price data found for ticker: {self.ticker}")
        prices = data["Close"]
        # Handle MultiIndex columns returned by newer yfinance versions
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.name = self.ticker
        return prices.dropna()

    def run(self, strategy):
        """
        Apply a strategy to the price data and return daily strategy returns.

        Args:
            strategy: Any object with a generate_signals(prices) method.

        Returns:
            pd.Series: Daily strategy returns.
        """
        signals = strategy.generate_signals(self.prices)

        # Daily percentage return of the asset
        daily_returns = self.prices.pct_change()

        # CRITICAL: shift signals by 1 to avoid look-ahead bias.
        # Signal is generated at end of day T, trade executes at start of day T+1.
        strategy_returns = signals.shift(1) * daily_returns

        return strategy_returns.dropna()

    def metrics(self, returns):
        """
        Compute key performance metrics for a strategy return series.

        Metrics explained:
        - Sharpe Ratio: risk-adjusted return. (mean / std) * sqrt(252).
          Annualised by sqrt(252) because there are ~252 trading days per year.
          A Sharpe > 1.0 is considered acceptable; > 2.0 is strong.
        - Max Drawdown: largest peak-to-trough decline. Critical for risk
          management — a strategy with great returns but -80% drawdown is
          unusable in practice.
        - Win Rate: percentage of days with positive returns.
        - P-Value: from a one-sample t-test against zero mean. Tests whether
          strategy returns are statistically significantly different from zero
          (i.e. from random noise). P < 0.05 means the result is unlikely to
          be due to chance at the 95% confidence level.

        Args:
            returns (pd.Series): Daily strategy returns.

        Returns:
            dict: Dictionary of computed metrics.
        """
        if returns.empty or returns.std() == 0:
            return {"Error": "Insufficient data to compute metrics."}

        # Annualised Sharpe Ratio
        # sqrt(252) converts daily Sharpe to annual — 252 trading days per year
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max Drawdown: worst peak-to-trough loss over the period
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate: fraction of days with positive return
        win_rate = (returns > 0).mean()

        # Total return: compound growth over the full period
        total_return = (1 + returns).prod() - 1

        # Annualised return
        n_years = len(returns) / 252
        annualised_return = (1 + total_return) ** (1 / n_years) - 1

        # Statistical significance: is the mean return different from 0?
        # H0: mean daily return = 0 (strategy has no edge)
        # H1: mean daily return != 0 (strategy has edge)
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        significant = p_value < 0.05

        return {
            "Ticker": self.ticker,
            "Total Return": f"{total_return:.2%}",
            "Annualised Return": f"{annualised_return:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "T-Statistic": f"{t_stat:.3f}",
            "P-Value": f"{p_value:.3f}",
            "Statistically Significant": "Yes ✓" if significant else "No ✗",
        }
