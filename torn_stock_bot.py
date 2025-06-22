#!/usr/bin/env python3
"""Simulated mean-reversion trading bot for Torn City stocks."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import requests

# Constants
API_KEY = "XLE1EjC6WkHeVrn7"
BUY_THRESH = 0.95
SELL_THRESH = 1.05
SIMULATION_ONLY = True

BASE_URL = "https://api.torn.com"
DATA_DIR = Path("torn_bot_data")
PRICES_DIR = DATA_DIR / "prices"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TRADES_FILE = DATA_DIR / "trades.csv"


@dataclass
class StockPrice:
    """Historical price entry."""

    date: str
    price: float


@dataclass
class Holding:
    """Portfolio holding."""

    shares: int
    avg_price: float


def torn_get(endpoint: str, selections: str) -> Dict:
    """Query the Torn API and return JSON data."""
    url = f"{BASE_URL}/{endpoint}/?selections={selections}&key={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error'].get('error')}")
    return data


def load_prices(symbol: str) -> List[StockPrice]:
    """Load the price history for a symbol."""
    path = PRICES_DIR / f"{symbol}.csv"
    prices: List[StockPrice] = []
    if not path.exists():
        return prices
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            prices.append(StockPrice(row[0], float(row[1])))
    return prices


def append_price(symbol: str, date: str, price: float) -> None:
    """Append today's price to history, keeping a rolling 7 day window."""
    prices = load_prices(symbol)
    if prices and prices[-1].date == date:
        return
    prices.append(StockPrice(date, price))
    prices = prices[-7:]
    path = PRICES_DIR / f"{symbol}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for p in prices:
            writer.writerow([p.date, f"{p.price:.2f}"])


def load_portfolio() -> Dict[str, Holding]:
    """Load portfolio from disk."""
    if not PORTFOLIO_FILE.exists():
        return {}
    with PORTFOLIO_FILE.open("r") as f:
        data = json.load(f)
    portfolio = {k: Holding(**v) for k, v in data.items()}
    return portfolio


def save_portfolio(portfolio: Dict[str, Holding]) -> None:
    """Save portfolio to disk."""
    DATA_DIR.mkdir(exist_ok=True)
    with PORTFOLIO_FILE.open("w") as f:
        json.dump({k: v.__dict__ for k, v in portfolio.items()}, f, indent=2)


def log_trade(action: str, symbol: str, shares: int, price: float, cash: float, pl: float) -> None:
    """Append a trade entry."""
    is_new = not TRADES_FILE.exists()
    with TRADES_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["date", "action", "symbol", "shares", "price", "cash_after", "realised_pl"])
        writer.writerow([
            datetime.utcnow().strftime("%Y-%m-%d"),
            action,
            symbol,
            shares,
            f"{price:.2f}",
            f"{cash:.2f}",
            f"{pl:.2f}",
        ])


def decide_trades() -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, Holding]], Dict[str, Holding]]:
    """Determine buy and sell candidates."""
    DATA_DIR.mkdir(exist_ok=True)
    PRICES_DIR.mkdir(exist_ok=True)
    stocks_data = torn_get("torn", "stocks")
    portfolio = load_portfolio()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    buy_list: List[Tuple[str, float, float]] = []
    sell_list: List[Tuple[str, float, Holding]] = []

    for stock in stocks_data.get("stocks", {}).values():
        symbol = stock.get("symbol") or stock.get("ticker")
        price = float(stock.get("current_price") or stock.get("price") or 0)
        if not symbol or price <= 0:
            continue
        append_price(symbol, today, price)
        history = [p.price for p in load_prices(symbol)]
        if len(history) < 7:
            continue
        avg7 = mean(history)
        vol = pstdev(history) / avg7 if avg7 else 0.0
        if symbol not in portfolio and price < BUY_THRESH * avg7:
            buy_list.append((symbol, price, vol))
        if symbol in portfolio and price > SELL_THRESH * avg7:
            sell_list.append((symbol, price, portfolio[symbol]))

    buy_list.sort(key=lambda x: x[2], reverse=True)
    return buy_list, sell_list, portfolio


def simulate_trades() -> None:
    """Main orchestration of the simulation."""
    buy_candidates, sell_candidates, portfolio = decide_trades()
    user_money = torn_get("user", "money")
    cash = float(user_money.get("money_onhand", 0))
    starting_cash = cash
    realised_pl = 0.0

    print(f"=== Torn Stock Bot — {datetime.utcnow().strftime('%Y-%m-%d')} (simulation mode) ===")
    print(f"Cash balance: ${cash:,.2f}")
    print(f"{len(buy_candidates)} buy-candidates | {len(sell_candidates)} sell-candidates")

    # Process sells first
    for symbol, price, holding in sell_candidates:
        proceeds = price * holding.shares * 0.999
        pl = proceeds - (holding.avg_price * holding.shares)
        cash += proceeds
        realised_pl += pl
        del portfolio[symbol]
        log_trade("SELL", symbol, holding.shares, price, cash, pl)
        print(f"SELL {holding.shares} {symbol} @ ${price:.2f} -> Cash ${cash:,.2f} (P/L {pl:+.2f})")

    # Allocate cash for buys
    if buy_candidates:
        cash_per_stock = cash / len(buy_candidates)
    else:
        cash_per_stock = 0.0

    for symbol, price, _ in buy_candidates:
        shares = int(math.floor(cash_per_stock / price))
        if shares <= 0:
            continue
        cost = shares * price
        cash -= cost
        portfolio[symbol] = Holding(shares, price)
        log_trade("BUY", symbol, shares, price, cash, 0.0)
        print(f"BUY {shares} {symbol} @ ${price:.2f} -> Cash ${cash:,.2f}")

    save_portfolio(portfolio)

    print(f"End-of-run cash: ${cash:,.2f}")
    print(f"Realised P/L today: ${realised_pl:,.2f}")
    print(f"Open positions: {len(portfolio)} — stored in {PORTFOLIO_FILE}")


if __name__ == "__main__":
    simulate_trades()
