"""
TRADING BOT IMPLEMENTATION
--------------------------------------
A Python trading bot that implements a Moving Average Crossover strategy,
backtests it against historical stock data, simulates trades with a
starting cash balance, and reports performance.

Strategy logic:
- Calculate a short-term moving average (SMA) and a long-term moving average
- BUY when the short SMA crosses above the long SMA (bullish signal / "golden cross")
- SELL when the short SMA crosses below the long SMA (bearish signal / "death cross")

Data source:
- Uses yfinance to pull real historical daily prices for a given stock ticker
- Falls back to generated synthetic price data automatically if yfinance
  or an internet connection isn't available, so the bot still runs and
  demonstrates the strategy end-to-end.

Run with: python trading_bot.py
Dependencies: pip install yfinance pandas matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
TICKER = "AAPL"
PERIOD_DAYS = 365          # how many days of historical data to analyze
SHORT_WINDOW = 20          # short-term moving average (days)
LONG_WINDOW = 50           # long-term moving average (days)
STARTING_CASH = 10000.0    # simulated starting balance ($)


# ---------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------
def fetch_price_data(ticker: str, days: int) -> pd.DataFrame:
    """
    Attempts to fetch real historical data using yfinance.
    Falls back to synthetic (randomly generated but realistic-looking)
    price data if yfinance is unavailable, so the bot always runs.
    """
    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=days)
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        data = data[["Close"]].rename(columns={"Close": "close"})
        data.index.name = "date"
        print(f"Loaded {len(data)} days of real data for {ticker} via yfinance.")
        return data
    except Exception as e:
        print(f"[Info] Could not fetch live data ({e}).")
        print("Falling back to simulated price data for demonstration.\n")
        return generate_synthetic_data(days)


def generate_synthetic_data(days: int, start_price: float = 150.0) -> pd.DataFrame:
    """Generates a random-walk price series that mimics real stock behavior."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")  # business days
    daily_returns = np.random.normal(loc=0.0005, scale=0.015, size=len(dates))
    prices = start_price * (1 + daily_returns).cumprod()
    df = pd.DataFrame({"close": prices}, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------
# STRATEGY
# ---------------------------------------------------------------------
def apply_strategy(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Adds moving averages and buy/sell signals to the dataframe."""
    df = df.copy()
    df["sma_short"] = df["close"].rolling(window=short_window).mean()
    df["sma_long"] = df["close"].rolling(window=long_window).mean()

    df["signal"] = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1   # bullish
    df.loc[df["sma_short"] <= df["sma_long"], "signal"] = -1  # bearish

    # A "trade" happens only when the signal changes (crossover event)
    df["position_change"] = df["signal"].diff()
    return df


# ---------------------------------------------------------------------
# BACKTEST / SIMULATED TRADING
# ---------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, starting_cash: float):
    cash = starting_cash
    shares_held = 0
    trade_log = []
    portfolio_values = []

    for date, row in df.iterrows():
        price = row["close"]
        change = row["position_change"]

        # BUY signal (crossed from bearish/neutral to bullish)
        if change == 2 or change == 1:
            if shares_held == 0 and cash > 0:
                shares_held = cash / price
                cash = 0.0
                trade_log.append({"date": date, "action": "BUY",
                                   "price": round(price, 2),
                                   "shares": round(shares_held, 4)})

        # SELL signal (crossed from bullish to bearish)
        elif change == -2 or change == -1:
            if shares_held > 0:
                cash = shares_held * price
                trade_log.append({"date": date, "action": "SELL",
                                   "price": round(price, 2),
                                   "cash_after": round(cash, 2)})
                shares_held = 0.0

        total_value = cash + shares_held * price
        portfolio_values.append(total_value)

    df["portfolio_value"] = portfolio_values

    # Liquidate any remaining position at the final price for a fair comparison
    final_price = df["close"].iloc[-1]
    final_value = cash + shares_held * final_price

    return trade_log, final_value, df


# ---------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------
def print_report(trade_log, starting_cash, final_value, df, ticker):
    print("=" * 55)
    print(f"TRADING BOT REPORT — {ticker}")
    print("=" * 55)
    print(f"Strategy: SMA({SHORT_WINDOW}) / SMA({LONG_WINDOW}) Crossover")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Starting cash: ${starting_cash:,.2f}")
    print(f"Total trades executed: {len(trade_log)}\n")

    print("--- Trade Log ---")
    if not trade_log:
        print("No trades were triggered in this period.")
    for t in trade_log:
        if t["action"] == "BUY":
            print(f"{t['date'].date()}  BUY   {t['shares']:.4f} shares @ ${t['price']}")
        else:
            print(f"{t['date'].date()}  SELL  -> cash: ${t['cash_after']}  @ ${t['price']}")

    profit = final_value - starting_cash
    pct_return = (profit / starting_cash) * 100

    # Buy & hold comparison (baseline)
    buy_hold_shares = starting_cash / df["close"].iloc[0]
    buy_hold_value = buy_hold_shares * df["close"].iloc[-1]
    buy_hold_return = ((buy_hold_value - starting_cash) / starting_cash) * 100

    print("\n--- Performance Summary ---")
    print(f"Final portfolio value:   ${final_value:,.2f}")
    print(f"Net profit / loss:       ${profit:,.2f}  ({pct_return:+.2f}%)")
    print(f"Buy & Hold comparison:   ${buy_hold_value:,.2f}  ({buy_hold_return:+.2f}%)")
    print("=" * 55)


def plot_results(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(df.index, df["close"], label="Close Price", color="#2c3e50", linewidth=1.2)
    ax1.plot(df.index, df["sma_short"], label=f"SMA {SHORT_WINDOW}", color="#3498db", linewidth=1)
    ax1.plot(df.index, df["sma_long"], label=f"SMA {LONG_WINDOW}", color="#e67e22", linewidth=1)

    buys = df[df["position_change"].isin([1, 2])]
    sells = df[df["position_change"].isin([-1, -2])]
    ax1.scatter(buys.index, buys["close"], marker="^", color="green", s=100, label="BUY", zorder=5)
    ax1.scatter(sells.index, sells["close"], marker="v", color="red", s=100, label="SELL", zorder=5)

    ax1.set_title(f"{ticker} — Price & Moving Average Crossover Signals")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["portfolio_value"], color="#27ae60", linewidth=1.5)
    ax2.set_title("Simulated Portfolio Value Over Time")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("trading_bot_results.png", dpi=150)
    print("\nChart saved as trading_bot_results.png")
    plt.show()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = fetch_price_data(TICKER, PERIOD_DAYS)
    df = apply_strategy(df, SHORT_WINDOW, LONG_WINDOW)
    trade_log, final_value, df = run_backtest(df, STARTING_CASH)
    print_report(trade_log, STARTING_CASH, final_value, df, TICKER)
    plot_results(df, TICKER)


if __name__ == "__main__":
    main()
