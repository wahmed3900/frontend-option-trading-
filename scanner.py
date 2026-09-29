"""
Scan logic shared by the API. Finds the put closest to a target delta on the
nearest expiration in a days-to-expiry window, via Yahoo Finance (yfinance).
"""
import math
import time
from datetime import date, datetime

import yfinance as yf

CACHE_SECONDS = 120
_cache = {}


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_delta(spot, strike, t_years, iv, rate):
    if iv <= 0 or t_years <= 0 or strike <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv**2) * t_years) / (iv * math.sqrt(t_years))
    return norm_cdf(d1) - 1


def pick_expiry(expirations, min_dte, max_dte):
    today = date.today()
    for exp in expirations:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if min_dte <= dte <= max_dte:
            return exp, dte
    return None, None


def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def scan_ticker(ticker, target_delta, min_dte, max_dte, rate=0.04):
    """Return one result dict, or raise ValueError with a readable reason."""
    key = (ticker, round(target_delta, 3), min_dte, max_dte, round(rate, 4))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    tk = yf.Ticker(ticker)
    hist = tk.history(period="1mo")
    if hist.empty:
        raise ValueError(f"{ticker}: no price data")
    closes = [round(_num(c), 2) for c in hist["Close"].tolist()]
    spot = closes[-1]

    expiry, dte = pick_expiry(list(tk.options), min_dte, max_dte)
    if expiry is None:
        raise ValueError(f"{ticker}: no expiration {min_dte}-{max_dte} days out")

    puts = tk.option_chain(expiry).puts.copy()
    puts = puts[(puts["strike"] < spot) & (puts["bid"] > 0) & (puts["impliedVolatility"] > 0.01)]
    if puts.empty:
        raise ValueError(f"{ticker}: no out-of-the-money puts with a bid")

    t_years = max(dte, 1) / 365
    puts["delta"] = puts.apply(
        lambda r: put_delta(spot, r["strike"], t_years, r["impliedVolatility"], rate), axis=1
    )
    puts = puts.dropna(subset=["delta"])
    puts["dist"] = (puts["delta"].abs() - target_delta).abs()
    best = puts.loc[puts["dist"].idxmin()]

    bid, ask, strike = _num(best["bid"]), _num(best["ask"]), _num(best["strike"])
    mid = (bid + ask) / 2 if ask > 0 else bid
    yld = bid / strike
    result = {
        "ticker": ticker,
        "contract": str(best["contractSymbol"]),
        "price": spot,
        "expiry": expiry,
        "dte": dte,
        "strike": strike,
        "delta": round(_num(best["delta"]), 3),
        "bid": bid,
        "mid": round(mid, 2),
        "iv": round(_num(best["impliedVolatility"]), 4),
        "premium": round(bid * 100, 2),
        "collateral": round(strike * 100, 2),
        "yield": yld,
        "annualized": yld * 365 / max(dte, 1),
        "otm": 1 - strike / spot,
        "change_1m": spot / closes[0] - 1 if closes[0] else 0.0,
        "trend": closes,
    }
    _cache[key] = (time.time(), result)
    return result
