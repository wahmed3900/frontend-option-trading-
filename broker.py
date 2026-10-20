"""
Order placement — Alpaca (default) or Robinhood via robin_stocks.

Keys (in .env or Render env vars):
  Alpaca:    ALPACA_API_KEY, ALPACA_SECRET_KEY
  Robinhood: RH_USERNAME, RH_PASSWORD
             RH_MFA_CODE  (only if your account has 2FA enabled)

Set BROKER=robinhood to use Robinhood; default is Alpaca.
"""
import os
import re

BROKER = os.getenv("BROKER", "alpaca").lower()  # "alpaca" or "robinhood"


# ---------- shared ----------
def round_price(price):
    """Options tick: $0.01 under $3, $0.05 at or above $3."""
    tick = 0.01 if price < 3 else 0.05
    return round(max(tick, round(price / tick) * tick), 2)


# ---------- Alpaca ----------
def _alpaca_client(paper=True):
    from alpaca.trading.client import TradingClient
    return TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper
    )


def _alpaca_buying_power(tc):
    acct = tc.get_account()
    bp = getattr(acct, "options_buying_power", None) or acct.buying_power
    return float(bp)


def _alpaca_sell_put(tc, contract_symbol, qty, limit_price):
    from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest
    order = LimitOrderRequest(
        symbol=contract_symbol,
        qty=qty,
        side=OrderSide.SELL,
        position_intent=PositionIntent.SELL_TO_OPEN,
        time_in_force=TimeInForce.DAY,
        limit_price=round_price(limit_price),
    )
    return tc.submit_order(order)


# ---------- Robinhood ----------
def _rh_login():
    """Login once per process (robin_stocks caches the session token)."""
    import robin_stocks.robinhood as rh
    username = os.environ["RH_USERNAME"]
    password = os.environ["RH_PASSWORD"]
    mfa = os.getenv("RH_MFA_CODE")  # leave unset if 2FA is off
    rh.login(username, password, mfa_code=mfa, store_session=True, expiresIn=86400)
    return rh


def _rh_buying_power():
    rh = _rh_login()
    profile = rh.profiles.load_account_profile()
    # options_buying_power is the cash available to secure CSPs
    bp = (profile.get("options_buying_power")
          or profile.get("buying_power")
          or "0")
    return float(bp)


def _rh_sell_put(contract_symbol, qty, limit_price):
    """
    contract_symbol is OCC format: e.g. NVDA261017P00100000
    Parses it, finds the Robinhood instrument, and places a sell-to-open limit order.
    """
    rh = _rh_login()

    # Parse OCC: TICKER YYMMDD [CP] STRIKE*1000
    m = re.fullmatch(r"([A-Z]{1,6})(\d{6})([CP])(\d{8})", contract_symbol)
    if not m:
        raise ValueError(f"Can't parse OCC symbol: {contract_symbol}")
    ticker, exp_str, _, strike_raw = m.groups()
    exp_date = f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:]}"
    strike = int(strike_raw) / 1000

    # Verify the option exists on Robinhood
    chains = rh.options.find_options_by_expiration_and_strike(
        ticker, exp_date, str(strike), "put"
    )
    if not chains:
        raise ValueError(
            f"Option not found on Robinhood: {ticker} {strike} put exp {exp_date}"
        )

    order = rh.orders.order_sell_option_limit(
        positionEffect="open",
        creditOrDebit="credit",
        price=round_price(limit_price),
        symbol=ticker,
        quantity=qty,
        expirationDate=exp_date,
        strike=strike,
        optionType="put",
    )
    return order


class _RhOrder:
    """Wraps a Robinhood order dict to look like an Alpaca order object."""
    def __init__(self, d):
        self.id = d.get("id", "unknown")
        self.status = d.get("state", "queued")


# ---------- public API (called by api.py) ----------
def client(paper=True):
    """Return a broker client, or None for Robinhood (stateless login)."""
    if BROKER == "robinhood":
        return None
    return _alpaca_client(paper=paper)


def buying_power(tc):
    if BROKER == "robinhood":
        return _rh_buying_power()
    return _alpaca_buying_power(tc)


def sell_put(tc, contract_symbol, qty, limit_price):
    if BROKER == "robinhood":
        result = _rh_sell_put(contract_symbol, qty, limit_price)
        return _RhOrder(result)
    return _alpaca_sell_put(tc, contract_symbol, qty, limit_price)
