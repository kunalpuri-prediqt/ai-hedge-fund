"""
UT Bot Alerts — simplified TradingView equivalent.

Usage (IPython / Jupyter):
    from ut_bot import analyze
    analyze("AAPL")
    analyze("AAPL", timeframe="weekly", atr_period=10, key_value=1)

Dependencies:
    pip install yfinance pandas numpy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "daily":   {"interval": "1d",  "period": "1y"},
    "weekly":  {"interval": "1wk", "period": "2y"},
    "monthly": {"interval": "1mo", "period": "5y"},
}


# ── Core: UT Bot trailing stop + signals ────────────────────────────────────

def ut_bot(df: pd.DataFrame, atr_period: int = 10, key_value: float = 1.0) -> pd.DataFrame:
    """
    Compute the UT Bot Alert (ATR trailing stop with buy/sell signals).

    This replicates the TradingView "UT Bot Alerts" indicator by @QuantNomad.
    Logic: ATR-based trailing stop that flips direction on price crossovers.
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values

    n = len(close)

    # ATR calculation (Wilder's smoothing)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[atr_period] = np.mean(tr[1:atr_period + 1])
    for i in range(atr_period + 1, n):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    n_loss = key_value * atr

    # Trailing stop
    trail = np.zeros(n)
    for i in range(1, n):
        if close[i] > trail[i - 1] and close[i - 1] > trail[i - 1]:
            trail[i] = max(trail[i - 1], close[i] - n_loss[i])
        elif close[i] < trail[i - 1] and close[i - 1] < trail[i - 1]:
            trail[i] = min(trail[i - 1], close[i] + n_loss[i])
        elif close[i] > trail[i - 1]:
            trail[i] = close[i] - n_loss[i]
        else:
            trail[i] = close[i] + n_loss[i]

    # Position: 1 = long, -1 = short, 0 = undecided
    pos = np.zeros(n, dtype=int)
    for i in range(1, n):
        if close[i] > trail[i] and close[i - 1] <= trail[i - 1]:
            pos[i] = 1
        elif close[i] < trail[i] and close[i - 1] >= trail[i - 1]:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    # Buy/sell signals (only on crossover bars)
    buy  = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if pos[i] == 1 and pos[i - 1] != 1:
            buy[i] = True
        if pos[i] == -1 and pos[i - 1] != -1:
            sell[i] = True

    df = df.copy()
    df["ATR"]       = atr
    df["Trail_Stop"] = trail
    df["Position"]  = pos
    df["Buy"]       = buy
    df["Sell"]      = sell

    return df


# ── Display ─────────────────────────────────────────────────────────────────

def _fmt_signal(row):
    if row["Buy"]:
        return "BUY"
    if row["Sell"]:
        return "SELL"
    return ""

def _print_report(ticker: str, tf: str, df: pd.DataFrame, atr_period: int, key_value: float):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Current state
    price = last["Close"]
    stop  = last["Trail_Stop"]
    atr   = last["ATR"]
    pos   = last["Position"]

    state = "LONG (above stop)" if pos == 1 else "SHORT (below stop)" if pos == -1 else "NEUTRAL"
    distance_pct = (price - stop) / price * 100

    print()
    print(f"{'=' * 60}")
    print(f"  UT Bot Alert  |  {ticker}  |  {tf}")
    print(f"  ATR Period: {atr_period}  |  Key Value: {key_value}")
    print(f"{'=' * 60}")
    print()
    print(f"  Price:          {price:>10.2f}")
    print(f"  Trailing Stop:  {stop:>10.2f}  ({distance_pct:+.1f}% away)")
    print(f"  ATR:            {atr:>10.2f}")
    print(f"  Current State:  {state}")
    print()

    # Latest signal
    if last["Buy"]:
        print(f"  ** BUY SIGNAL on {df.index[-1].strftime('%Y-%m-%d')} **")
    elif last["Sell"]:
        print(f"  ** SELL SIGNAL on {df.index[-1].strftime('%Y-%m-%d')} **")
    else:
        # Find most recent signal
        signals = df[df["Buy"] | df["Sell"]].tail(1)
        if not signals.empty:
            s = signals.iloc[-1]
            sig_type = "BUY" if s["Buy"] else "SELL"
            sig_date = signals.index[-1].strftime("%Y-%m-%d")
            sig_price = s["Close"]
            bars_ago = len(df) - df.index.get_loc(signals.index[-1]) - 1
            print(f"  Last signal:    {sig_type} on {sig_date} @ {sig_price:.2f} ({bars_ago} bars ago)")

    print()

    # Recent signals table
    recent = df[df["Buy"] | df["Sell"]].tail(8)
    if not recent.empty:
        print(f"  Recent Signals:")
        print(f"  {'Date':<12} {'Signal':<6} {'Price':>10} {'Stop':>10}")
        print(f"  {'-' * 42}")
        for idx, row in recent.iterrows():
            sig = "BUY" if row["Buy"] else "SELL"
            print(f"  {idx.strftime('%Y-%m-%d'):<12} {sig:<6} {row['Close']:>10.2f} {row['Trail_Stop']:>10.2f}")
        print()

    # Quick context
    print(f"  {'─' * 40}")
    print(f"  Stop acts as exit:  {'Close below' if pos == 1 else 'Close above'} {stop:.2f}")
    print(f"{'=' * 60}")
    print()


# ── Main entry point ────────────────────────────────────────────────────────

def analyze(
    ticker: str,
    timeframe: str = "daily",
    atr_period: int = 10,
    key_value: float = 1.0,
    show_data: bool = False,
):
    """
    Pull data and run UT Bot Alerts for a ticker.

    Args:
        ticker:     Stock symbol (e.g. "AAPL", "MSFT")
        timeframe:  "daily", "weekly", or "monthly"
        atr_period: ATR lookback period (default 10)
        key_value:  ATR multiplier / sensitivity (default 1.0, higher = wider stops)
        show_data:  If True, return the full DataFrame instead of just printing

    Examples:
        analyze("AAPL")
        analyze("TSLA", timeframe="weekly", key_value=2)
        df = analyze("MSFT", show_data=True)
    """
    tf = timeframe.lower()
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"timeframe must be one of {list(TIMEFRAME_MAP.keys())}")

    config = TIMEFRAME_MAP[tf]
    df = yf.download(ticker, period=config["period"], interval=config["interval"], progress=False)

    if df.empty:
        print(f"No data returned for {ticker}")
        return None

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = ut_bot(df, atr_period=atr_period, key_value=key_value)
    _print_report(ticker, tf, df, atr_period, key_value)

    if show_data:
        return df


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    analyze(ticker)
