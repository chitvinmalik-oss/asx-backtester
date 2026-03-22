"""
strategy.py
Defines trading strategy logic. Currently implements a Simple Moving
Average (SMA) crossover strategy.
"""

import numpy as np
import pandas as pd


class SMACrossoverStrategy:
    """
    Simple Moving Average (SMA) Crossover Strategy.

    Generates a LONG signal (+1) when the fast SMA crosses above the slow SMA,
    and a SHORT signal (-1) when it crosses below. This captures momentum:
    when short-term average price exceeds long-term average, the asset is
    trending upward.

    Args:
        fast_window (int): Lookback period for the fast (short-term) SMA.
        slow_window (int): Lookback period for the slow (long-term) SMA.
    """

    def __init__(self, fast_window=20, slow_window=50):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window.")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices):
        """
        Compute daily position signals from a price series.

        Args:
            prices (pd.Series): Daily closing prices indexed by date.

        Returns:
            pd.Series: Signal series — +1 (long) or -1 (short) per day.
        """
        fast_sma = prices.rolling(self.fast_window).mean()
        slow_sma = prices.rolling(self.slow_window).mean()

        # +1 when fast SMA is above slow SMA (uptrend), -1 otherwise
        signals = np.where(fast_sma > slow_sma, 1, -1)
        return pd.Series(signals, index=prices.index, name="signal")
