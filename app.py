"""
Weekly put scanner — the whole app in one file (page + API), Flask on Vercel.

Vercel: set Application Preset to "Flask". Add these in
Settings → Environment Variables, then redeploy:
  APP_PASSWORD                      (required for orders and sending notifications)
  GEMINI_API_KEY and/or ANTHROPIC_API_KEY
  OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY, OMNIROUTE_MODEL (optional gateway)
  ALPACA_API_KEY, ALPACA_SECRET_KEY (paper keys to start)
  MAX_CONTRACTS=1, MAX_COLLATERAL_PCT=0.25, LIVE_TRADING=false
  EMAIL_FROM, EMAIL_APP_PASSWORD, ALERT_EMAIL (optional email results)
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, ALERT_PHONE (optional SMS)

The per-process request limiter is a best-effort safeguard. Use a shared rate-limit
store at the edge when deploying multiple serverless instances.

On your Mac:  APP_PASSWORD=test python app.py  → http://localhost:5000
"""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from datetime import date, datetime, timezone
from functools import wraps

from flask import Flask, Response, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
try:
    MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "1"))
    MAX_COLLATERAL_PCT = float(os.getenv("MAX_COLLATERAL_PCT", "0.25"))
except ValueError as exc:
    raise RuntimeError("MAX_CONTRACTS and MAX_COLLATERAL_PCT must be valid numbers.") from exc
if not 1 <= MAX_CONTRACTS <= 100:
    raise RuntimeError("MAX_CONTRACTS must be between 1 and 100.")
if not math.isfinite(MAX_COLLATERAL_PCT) or not 0 < MAX_COLLATERAL_PCT <= 1:
    raise RuntimeError("MAX_COLLATERAL_PCT must be greater than 0 and at most 1.")
LIVE_ALLOWED = os.getenv("LIVE_TRADING", "").lower() == "true"
STALE_SECONDS = 300
RATE_WINDOW_SECONDS = 60
RATE_LIMITS = {"scan": 6, "analyze": 4, "news": 6, "message": 3, "order": 5}
_rate_lock = threading.Lock()
_rate_hits = {}
_news_cache = {}


def rate_limited(bucket):
    """Best-effort per-process throttling; use a shared limiter for multi-instance hosting."""
    now = time.time()
    key = (bucket, request.remote_addr or "unknown")
    with _rate_lock:
        hits = [stamp for stamp in _rate_hits.get(key, []) if now - stamp < RATE_WINDOW_SECONDS]
        # Bound the in-process limiter state on long-running hosts.
        if len(_rate_hits) > 5000:
            for old_key, old_hits in list(_rate_hits.items()):
                if not old_hits or now - old_hits[-1] >= RATE_WINDOW_SECONDS:
                    _rate_hits.pop(old_key, None)
        if len(hits) >= RATE_LIMITS[bucket]:
            _rate_hits[key] = hits
            return True
        hits.append(now)
        _rate_hits[key] = hits
    return False


# ---------------- helpers ----------------
def fail(status, message):
    return jsonify({"detail": message}), status


def require_password(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not APP_PASSWORD:
            return fail(503, "APP_PASSWORD is not set in Vercel.")
        if rate_limited("order"):
            return fail(429, "Too many protected requests. Wait a minute and try again.")
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), APP_PASSWORD.encode()):
            return fail(401, "Wrong password.")
        return fn(*args, **kwargs)
    return wrapper


def sign(contract, ts):
    """Proof that a contract came from a scan at time ts (no database needed)."""
    msg = f"{contract}|{ts}".encode()
    return hmac.new(APP_PASSWORD.encode(), msg, hashlib.sha256).hexdigest()


def num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


# ---------------- scanner ----------------
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


def scan_ticker(ticker, target, min_dte, max_dte, rate=0.04):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="1mo")
    if hist.empty:
        raise ValueError(f"{ticker}: no price data")
    closes = [round(num(c), 2) for c in hist["Close"].tolist()]
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
    puts["dist"] = (puts["delta"].abs() - target).abs()
    best = puts.loc[puts["dist"].idxmin()]

    bid, ask, strike = num(best["bid"]), num(best["ask"]), num(best["strike"])
    mid = (bid + ask) / 2 if ask > 0 else bid
    yld = bid / strike
    contract = str(best["contractSymbol"])
    ts = int(time.time())
    return {
        "ticker": ticker, "contract": contract, "price": spot, "expiry": expiry, "dte": dte,
        "strike": strike, "delta": round(num(best["delta"]), 3), "bid": bid, "mid": round(mid, 2),
        "iv": round(num(best["impliedVolatility"]), 4), "premium": round(bid * 100, 2),
        "collateral": round(strike * 100, 2), "yield": yld, "annualized": yld * 365 / max(dte, 1),
        "otm": 1 - strike / spot, "change_1m": spot / closes[0] - 1 if closes[0] else 0.0,
        "trend": closes, "quote_asof": str(hist.index[-1]),
        "ticket": {"ts": ts, "sig": sign(contract, ts)},
    }


# ---------------- AI ----------------
AI_SYSTEM = """You review cash-secured put candidates for a weekly options seller.
Weigh premium yield against risk: implied volatility, distance out of the money,
the 1-month trend, days to expiry, and the extra gap risk of leveraged ETFs.
Respond ONLY with JSON, no markdown, in exactly this shape:
{"pick": "<TICKER or NONE>", "confidence": "low" | "medium" | "high",
 "summary": "<2-3 sentences>", "risks": ["..."], "per_ticker": {"<TICKER>": "<one line>"}}
Only pick tickers present in the data."""


def ask_gemini(prompt, model):
    from google import genai
    from google.genai import types
    client = genai.Client()  # reads GEMINI_API_KEY
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=AI_SYSTEM, response_mime_type="application/json"),
    )
    return resp.text


def ask_claude(prompt, model):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model, max_tokens=1000, system=AI_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_omniroute(prompt, model):
    """OmniRoute speaks the OpenAI chat completions API."""
    import requests
    if not model:
        raise ValueError("set OMNIROUTE_MODEL in Vercel or type a model name")
    base = os.environ["OMNIROUTE_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    headers = {}
    if os.getenv("OMNIROUTE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OMNIROUTE_API_KEY']}"
    resp = requests.post(f"{base}/chat/completions", headers=headers, timeout=60, json={
        "model": model, "max_tokens": 1000,
        "messages": [{"role": "system", "content": AI_SYSTEM}, {"role": "user", "content": prompt}],
    })
    if not resp.ok:
        raise ValueError(f"OmniRoute returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"] or ""


def providers():
    p = {}
    if os.getenv("GEMINI_API_KEY"):
        p["Gemini"] = (ask_gemini, "gemini-3.8-flash")
    if os.getenv("ANTHROPIC_API_KEY"):
        p["Claude"] = (ask_claude, "claude-haiku-4-5")
    if os.getenv("OMNIROUTE_BASE_URL"):
        p["OmniRoute"] = (ask_omniroute, os.getenv("OMNIROUTE_MODEL", ""))
    return p


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


# ---------------- broker (Alpaca) ----------------
def alpaca(paper=True):
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper)


def round_price(p):
    tick = 0.01 if p < 3 else 0.05
    return round(max(tick, round(p / tick) * tick), 2)


# ---------------- routes ----------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/config.js")
def client_config():
    # The Flask-served page uses same-origin API routes. A separate static host
    # can keep its own config.js and point API_URL at the backend's /api prefix.
    return Response('window.API_URL = "";', mimetype="application/javascript")


@app.after_request
def add_cors_headers(response):
    origin = os.getenv("FRONTEND_ORIGIN", "*").strip()
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Vary"] = "Origin"
    return response


@app.get("/api/config")
def config():
    return jsonify({
        "providers": [{"name": n, "default_model": m} for n, (_, m) in providers().items()],
        "max_contracts": MAX_CONTRACTS,
        "max_collateral_pct": MAX_COLLATERAL_PCT,
        "live_allowed": LIVE_ALLOWED,
        "stale_seconds": STALE_SECONDS,
    })


@app.post("/api/scan")
def scan():
    if rate_limited("scan"):
        return fail(429, "Scan limit reached. Wait a minute and try again.")
    body = request.get_json(silent=True) or {}
    raw_tickers = body.get("tickers", [])
    if not isinstance(raw_tickers, list):
        return fail(400, "Tickers must be a list.")
    tickers = list(dict.fromkeys(str(t).strip().upper() for t in raw_tickers if str(t).strip()))[:12]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker) for ticker in tickers):
        return fail(400, "Use valid ticker symbols (letters, numbers, dots, or hyphens).")
    if not tickers:
        return fail(400, "Enter at least one ticker.")
    delta = min(max(num(body.get("delta"), 0.15), 0.05), 0.40)
    min_dte = int(min(max(num(body.get("min_dte"), 4), 0), 60))
    max_dte = int(min(max(num(body.get("max_dte"), 10), 1), 60))
    if min_dte > max_dte:
        return fail(400, "Minimum days out cannot exceed maximum days out.")

    results, errors = [], []
    for t in tickers:
        try:
            results.append(scan_ticker(t, delta, min_dte, max_dte))
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{t}: {e}")
    results.sort(key=lambda r: r["yield"], reverse=True)
    return jsonify({"results": results, "errors": errors, "scanned_at": time.time()})


@app.post("/api/analyze")
def analyze():
    if rate_limited("analyze"):
        return fail(429, "AI review limit reached. Wait a minute and try again.")
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return fail(400, "Invalid request body.")
    p = providers()
    name, model = body.get("provider"), str(body.get("model") or "")[:100]
    if name not in p:
        return fail(400, "That AI provider has no key set in Vercel.")
    submitted = body.get("rows") or []
    if not isinstance(submitted, list):
        return fail(400, "Candidates must be a list.")
    rows = [{k: v for k, v in r.items() if k not in ("contract", "trend", "ticket")}
            for r in submitted[:12] if isinstance(r, dict)]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", str(r.get("ticker", ""))) for r in rows):
        return fail(400, "Invalid ticker in candidates.")
    if not rows:
        return fail(400, "Scan first.")
    fn, default_model = p[name]
    if model and model != default_model:
        return fail(400, "Only the configured default model is allowed.")
    try:
        result = parse_json(fn("Candidates (yield = bid / strike):\n" + json.dumps(rows, indent=2), model or default_model))
        if not isinstance(result, dict):
            raise ValueError("AI response must be a JSON object.")
        if not isinstance(result.get("risks", []), list) or not isinstance(result.get("per_ticker", {}), dict):
            raise ValueError("AI response has an invalid shape.")
    except Exception as e:
        return fail(502, f"AI request failed: {e}")
    if result.get("pick") not in {r["ticker"] for r in rows}:
        result["pick"] = "NONE"
    return jsonify(result)


@app.post("/api/order")
@require_password
def order():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return fail(400, "Invalid request body.")
    contract = str(body.get("contract", ""))
    raw_qty = num(body.get("qty"), 0)
    if not raw_qty.is_integer():
        return fail(400, "Contracts must be a whole number.")
    qty = int(raw_qty)
    limit = num(body.get("limit"), 0)
    account = body.get("account", "paper")
    ticket = body.get("ticket") or {}
    if not isinstance(ticket, dict):
        return fail(400, "Invalid scan ticket. Scan again.")

    if account not in ("paper", "live"):
        return fail(400, "Unknown account.")
    if account == "live" and not LIVE_ALLOWED:
        return fail(403, "Live trading is turned off.")
    if not 1 <= qty <= MAX_CONTRACTS:
        return fail(400, f"Contracts must be 1 to {MAX_CONTRACTS}.")
    if limit <= 0:
        return fail(400, "Enter a limit price.")
    ts = int(num(ticket.get("ts"), 0))
    if not re.fullmatch(r"[A-Z]{1,6}\d{6}P\d{8}", contract) or not hmac.compare_digest(
        str(ticket.get("sig", "")), sign(contract, ts)
    ):
        return fail(400, "Contract isn't from a scan. Scan again.")
    age = time.time() - ts
    if age < 0 or age > STALE_SECONDS:
        return fail(409, "Quotes are over 5 minutes old. Scan again.")
    if account == "live" and body.get("manual_quote_confirmed") is not True:
        return fail(400, "Confirm that you checked the current quote with your broker.")

    strike = int(contract[-8:]) / 1000  # OCC symbol ends with strike × 1000
    collateral = strike * 100 * qty
    try:
        tc = alpaca(paper=(account == "paper"))
        acct = tc.get_account()
        bp = num(getattr(acct, "options_buying_power", None) or acct.buying_power)
    except KeyError:
        return fail(503, "Alpaca keys are not set in Vercel.")
    except Exception as e:
        return fail(502, f"Couldn't reach Alpaca: {e}")
    if collateral > bp * MAX_COLLATERAL_PCT:
        return fail(400, f"Blocked: ${collateral:,.0f} collateral is over "
                         f"{MAX_COLLATERAL_PCT:.0%} of ${bp:,.0f} buying power.")
    try:
        from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        client_order_id = hmac.new(
            APP_PASSWORD.encode(),
            f"{contract}|{ts}|{account}|{qty}|{round_price(limit)}".encode(),
            hashlib.sha256,
        ).hexdigest()[:48]
        existing = None
        try:
            existing = tc.get_order_by_client_id(client_order_id)
        except Exception:
            existing = None
        if existing:
            return jsonify({"id": str(existing.id), "status": str(existing.status), "account": account, "duplicate": True})
        o = tc.submit_order(LimitOrderRequest(
            symbol=contract, qty=qty, side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
            time_in_force=TimeInForce.DAY, limit_price=round_price(limit),
            client_order_id=client_order_id,
        ))
    except Exception as e:
        return fail(502, f"Order rejected: {e}")
    return jsonify({"id": str(o.id), "status": str(o.status), "account": account})


# ---------------- market news and notifications ----------------
@app.get("/api/news")
def news():
    if rate_limited("news"):
        return fail(429, "News refresh limit reached. Wait a minute and try again.")
    import yfinance as yf
    raw = request.args.get("tickers", "SPY,QQQ,NVDA")
    tickers = list(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))[:10]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker) for ticker in tickers):
        return fail(400, "Invalid ticker symbol.")
    cache_key = tuple(tickers)
    cached = _news_cache.get(cache_key)
    if cached and time.time() - cached[0] < 300:
        return jsonify(cached[1])
    seen, items, errors = set(), [], []
    for ticker in tickers:
        try:
            for item in (yf.Ticker(ticker).news or [])[:5]:
                content = item.get("content") or {}
                uid = item.get("id") or item.get("uuid") or item.get("link") or content.get("id")
                title = content.get("title") or item.get("title") or ""
                link = ((content.get("canonicalUrl") or {}).get("url") or item.get("link") or item.get("url") or "")
                parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(link)
                if not uid or uid in seen or not title or parsed.scheme != "https" or not parsed.hostname:
                    continue
                seen.add(uid)
                publisher = (content.get("provider") or {}).get("displayName") or item.get("publisher") or ""
                published = content.get("pubDate") or item.get("providerPublishTime") or 0
                if isinstance(published, str):
                    try:
                        published = int(datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp())
                    except ValueError:
                        published = 0
                items.append({"title": str(title)[:500], "url": link, "source": str(publisher)[:120],
                              "published": int(published or 0), "ticker": ticker})
        except Exception:
            errors.append(ticker)
    items.sort(key=lambda item: item["published"], reverse=True)
    response = {"items": items[:25], "errors": errors}
    _news_cache[cache_key] = (time.time(), response)
    if len(_news_cache) > 256:
        oldest = min(_news_cache, key=lambda key: _news_cache[key][0])
        _news_cache.pop(oldest, None)
    return jsonify(response)


def _authorized_message():
    if not APP_PASSWORD:
        return fail(503, "APP_PASSWORD is not set.")
    if rate_limited("message"):
        return fail(429, "Message limit reached. Wait a minute and try again.")
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token.encode(), APP_PASSWORD.encode()):
        return fail(401, "Wrong password.")
    return None


def _validated_results(body):
    rows = body.get("results")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("Provide between 1 and 12 scan results.")
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid scan result.")
        ticker = str(row.get("ticker", "")).upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker):
            raise ValueError("Invalid ticker in scan result.")
        values = {}
        for field in ("strike", "annualized", "yield", "otm", "bid", "premium", "collateral", "delta", "iv"):
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError
                values[field] = value
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Invalid {field} for {ticker}.")
        expiry = str(row.get("expiry", ""))
        try:
            datetime.strptime(expiry, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid expiry for {ticker}.")
        clean.append({"ticker": ticker, "expiry": expiry, "dte": int(num(row.get("dte"))), **values})
    return clean


@app.post("/api/email-results")
def email_results():
    auth_error = _authorized_message()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}
    try:
        rows = _validated_results(body)
    except ValueError as exc:
        return fail(400, str(exc))
    recipients = [x.strip() for x in os.getenv("ALERT_EMAIL", "").split(",") if x.strip()]
    if not recipients or len(recipients) > 5:
        return fail(503, "Configure up to five ALERT_EMAIL recipients.")
    from email.message import EmailMessage
    import smtplib
    sender, password = os.getenv("EMAIL_FROM", ""), os.getenv("EMAIL_APP_PASSWORD", "")
    if not sender or not password:
        return fail(503, "EMAIL_FROM and EMAIL_APP_PASSWORD must be configured.")
    html_rows = "".join("<tr>" + "".join(f"<td>{__import__('html').escape(str(value))}</td>" for value in
        (r['ticker'], r['expiry'], f"${r['strike']:.2f}", f"${r['bid']:.2f}", f"{r['annualized']*100:.1f}%")) + "</tr>" for r in rows)
    message = EmailMessage()
    message["Subject"] = f"Put Scanner Results — {date.today().isoformat()}"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content("Put scanner results are attached in the HTML version.")
    message.add_alternative(f"<h2>Weekly Put Scanner</h2><table><thead><tr><th>Ticker</th><th>Expiry</th><th>Strike</th><th>Bid</th><th>Annualized yield</th></tr></thead><tbody>{html_rows}</tbody></table><p>Not financial advice.</p>", subtype="html")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message, to_addrs=recipients)
    except Exception:
        app.logger.exception("Email delivery failed")
        return fail(502, "Email delivery failed. Check server logs.")
    return jsonify({"sent": len(recipients)})


@app.post("/api/text-results")
def text_results():
    auth_error = _authorized_message()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}
    try:
        rows = _validated_results(body)
    except ValueError as exc:
        return fail(400, str(exc))
    recipients = [x.strip() for x in os.getenv("ALERT_PHONE", "").split(",") if x.strip()]
    if not recipients or len(recipients) > 5 or any(not re.fullmatch(r"\+[1-9]\d{7,14}", x) for x in recipients):
        return fail(503, "Configure up to five valid ALERT_PHONE recipients in E.164 format.")
    sid, token, sender = (os.getenv(k, "") for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))
    if not sid or not token or not sender:
        return fail(503, "Twilio credentials and TWILIO_FROM must be configured.")
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        lines = ["Put Scanner — Not financial advice"]
        for row in rows[:4]:
            lines.append(f"{row['ticker']} ${row['strike']:.2f} put {row['expiry']} bid ${row['bid']:.2f}")
        for recipient in recipients:
            client.messages.create(body="\n".join(lines), from_=sender, to=recipient)
    except Exception:
        app.logger.exception("SMS delivery failed")
        return fail(502, "SMS delivery failed. Check server logs.")
    return jsonify({"sent": len(recipients)})


# ---------------- page ----------------
@app.get("/")
def home():
    page = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    return Response(page, mimetype="text/html")


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Weekly put scanner</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📉</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #EEF1F4; --surface: #FFFFFF; --ink: #14213D; --muted: #5C677D;
  --rule: #C9D1DC; --accent: #0F7B6C; --accent-soft: #D5EDE9;
  --bar: #9AA6BC; --risk: #B3261E; --risk-soft: #F6DEDC;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0E1626; --surface: #16213A; --ink: #E8EDF5; --muted: #9AA6BC;
    --rule: #2A3754; --accent: #3FB8A3; --accent-soft: #143A36;
    --bar: #4A5878; --risk: #F08A80; --risk-soft: #3A1E1E;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font: 400 16px/1.5 "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  font-variant-numeric: tabular-nums;
  padding: env(safe-area-inset-top, 0) env(safe-area-inset-right, 0) env(safe-area-inset-bottom, 0) env(safe-area-inset-left, 0);
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 20px; }
h2 { font-size: 1rem; font-weight: 600; margin: 0 0 12px; }
.muted { color: var(--muted); }
.hidden { display: none !important; }

/* controls */
label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: 4px; }
input, select {
  font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px; width: 100%;
}
input:focus-visible, select:focus-visible, button:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
button {
  font: inherit; font-weight: 600; cursor: pointer; border-radius: 6px;
  padding: 9px 16px; border: 1px solid var(--ink); background: var(--ink); color: var(--bg);
}
button.secondary { background: transparent; color: var(--ink); }
button:disabled { opacity: .4; cursor: not-allowed; }

.settings {
  display: grid; gap: 12px; align-items: end;
  grid-template-columns: 2fr 1fr 1fr 1fr auto;
  padding-bottom: 24px; border-bottom: 1px solid var(--rule);
}
@media (max-width: 720px) { .settings { grid-template-columns: 1fr 1fr; } .settings .tick { grid-column: 1 / -1; } }

/* verdict */
.verdict { padding: 40px 0 8px; }
.verdict .big { font-size: clamp(2rem, 5.5vw, 3.4rem); font-weight: 800; line-height: 1.08; letter-spacing: -.02em; margin: 0; }
.verdict .sub { font-size: 1.1rem; color: var(--muted); margin: 12px 0 0; max-width: 62ch; }

/* ranked bars */
.bars { margin: 32px 0 40px; }
.bar-row { display: grid; grid-template-columns: 64px 1fr 72px; gap: 14px; align-items: center; padding: 6px 0; }
.bar-row .t { font-weight: 600; }
.track { height: 28px; background: transparent; }
.fill { height: 100%; background: var(--bar); border-radius: 0 4px 4px 0; transition: width .5s ease; }
.bar-row.win .fill { background: var(--accent); }
.bar-row .v { text-align: right; font-weight: 600; }
.bar-row.win .v { color: var(--accent); }
@media (prefers-reduced-motion: reduce) { .fill { transition: none; } }

/* table */
.table-wrap { overflow-x: auto; border-top: 1px solid var(--rule); }
table { border-collapse: collapse; width: 100%; min-width: 720px; font-size: .92rem; }
th, td { padding: 10px 8px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font-weight: 500; color: var(--muted); }
th:first-child, td:first-child { text-align: left; }
td svg { vertical-align: middle; }
a.trade {
  display: inline-block; font-weight: 600; font-size: .85rem; text-decoration: none;
  padding: 5px 10px; border-radius: 6px; border: 1px solid var(--ink); color: var(--ink);
}
a.trade:hover { background: var(--ink); color: var(--bg); }
a.trade:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* panels */
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 40px; }
@media (max-width: 820px) { .panels { grid-template-columns: 1fr; } }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 20px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.pick { font-size: 1.4rem; font-weight: 800; margin: 16px 0 4px; }
.risks { margin: 8px 0 0; padding-left: 18px; }
.risks li { margin-bottom: 4px; }
.notes { margin-top: 12px; font-size: .9rem; color: var(--muted); }
.ticket-summary { background: var(--bg); border-radius: 6px; padding: 12px; margin: 12px 0; font-size: .95rem; }
.check { display: flex; gap: 10px; align-items: flex-start; font-size: .92rem; color: var(--ink); margin: 12px 0; }
.check input { width: auto; margin-top: 4px; }
.seg { display: flex; gap: 8px; }
.seg label { display: flex; gap: 6px; align-items: center; color: var(--ink); margin: 0; font-size: .95rem; }
.seg input { width: auto; }

.msg { border-radius: 6px; padding: 10px 12px; margin: 12px 0 0; font-size: .92rem; }
.msg.err { background: var(--risk-soft); color: var(--risk); }
.msg.ok { background: var(--accent-soft); color: var(--accent); }
.msg.info { background: var(--surface); border: 1px solid var(--rule); color: var(--muted); }


/* manual order at another broker */
.manual { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--rule); }
.manual h3 { font-size: .95rem; font-weight: 600; margin: 0 0 4px; }
.manual dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; margin: 12px 0; font-size: .95rem; }
.manual dt { color: var(--muted); }
.manual dd { margin: 0; font-weight: 600; }


.news-section { margin-top: 40px; }
.news-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
.news-header h2 { margin:0; }
.news-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }
.news-card { display:block; background:var(--surface); border:1px solid var(--rule); border-radius:9px; padding:14px; color:inherit; text-decoration:none; }
.news-card:hover { border-color:var(--accent); }
.news-meta,.news-source { color:var(--muted); font-size:.82rem; }
.news-badge { color:var(--accent); font-weight:600; margin-right:8px; }
.news-title { font-weight:600; margin:8px 0; }
.news-empty { color:var(--muted); }
.message-bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:20px; }
#message-status { color:var(--muted); font-size:.9rem; }

footer { margin-top: 48px; font-size: .85rem; color: var(--muted); max-width: 70ch; }
</style>
</head>
<body>

<main id="app" class="wrap">
  <h1>Weekly put scanner</h1>

  <form id="scan-form" class="settings">
    <div class="tick"><label for="tickers">Tickers</label><input id="tickers" value="NVDA, NVDL, NVDX, SOXL, TQQQ, SPXL"></div>
    <div><label for="delta">Target delta</label><input id="delta" type="number" min="0.05" max="0.40" step="0.01" value="0.15"></div>
    <div><label for="min-dte">Min days out</label><input id="min-dte" type="number" min="0" max="60" value="4"></div>
    <div><label for="max-dte">Max days out</label><input id="max-dte" type="number" min="1" max="60" value="10"></div>
    <button type="submit" id="scan-btn">Scan</button>
  </form>
  <div id="scan-msg"></div>

  <section id="results" class="hidden">
    <div class="verdict">
      <p class="big" id="verdict-big"></p>
      <p class="sub" id="verdict-sub"></p>
    </div>

    <div class="bars" id="bars" aria-label="Weekly premium as a percent of strike, highest first"></div>

    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Ticker</th><th>Price</th><th>Strike</th><th>Expires</th><th>Delta</th><th>Bid</th>
          <th>IV</th><th>Below price</th><th>Annualized</th><th>1 month</th><th>Trade</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>

    <div class="panels">
      <div class="panel">
        <h2>AI review</h2>
        <div class="row">
          <div><label for="provider">Provider</label><select id="provider"></select></div>
          <div><label for="model">Model</label><input id="model"></div>
        </div>
        <button id="analyze-btn" class="secondary" type="button">Review with AI</button>
        <div id="ai-out"></div>
      </div>

      <div class="panel">
        <h2>Order ticket</h2>
        <div class="row">
          <div><label for="o-ticker">Sell put on</label><select id="o-ticker"></select></div>
          <div><label for="o-qty">Contracts</label><input id="o-qty" type="number" min="1" value="1"></div>
        </div>
        <div class="row">
          <div><label for="o-limit">Limit price per share</label><input id="o-limit" type="number" min="0.01" step="0.01"></div>
          <div><label>Account</label>
            <div class="seg" id="o-account">
              <label><input type="radio" name="acct" value="paper" checked> Paper</label>
              <label id="live-opt" class="hidden"><input type="radio" name="acct" value="live"> Live</label>
            </div>
          </div>
        </div>
        <div class="ticket-summary" id="o-summary"></div>
        <div><label for="o-pw">Order password</label><input id="o-pw" type="password" autocomplete="current-password"></div>
        <label class="check"><input type="checkbox" id="o-confirm"> I've checked this against my broker and want to place it.</label>
        <button id="order-btn" type="button" disabled>Place order</button>
        <div id="order-msg"></div>

        <div class="manual">
          <h3>How to place this at Robinhood / Fidelity</h3>
          <p class="muted" style="margin:0;font-size:.9rem">Open the ticker's options chain in their app and enter these. Needs options approval for cash-secured puts.</p>
          <dl id="m-fields"></dl>
          <button id="m-copy" class="secondary" type="button">Copy order details</button>
          <div id="m-msg"></div>
        </div>
      </div>
    </div>
  </section>


  <section class="news-section" aria-labelledby="news-heading">
    <div class="news-header"><h2 id="news-heading">Market News</h2><button id="news-btn" class="secondary" type="button">Load News</button></div>
    <div id="news-out" aria-live="polite"></div>
  </section>
  <section class="message-bar hidden" id="message-bar" aria-label="Send scan results">
    <button id="email-btn" class="secondary" type="button">Email my configured recipients</button>
    <button id="text-btn" class="secondary" type="button">Text my configured recipients</button>
    <span id="message-status" role="status"></span>
  </section>
  <footer>
    Data from Yahoo Finance, delayed about 15 minutes. Delta is calculated from Yahoo's implied volatility.
    The AI only reviews; strike, size, and price come from the scan and your inputs.
    Leveraged ETFs pay more because they can gap through the strike. Not financial advice.
    <br>Need a brokerage account? <a href="https://www.fidelity.com/open-account/overview" target="_blank" rel="noopener noreferrer" style="color:inherit">Open one at Fidelity ↗</a>
  </footer>
</main>

<script>
const API = "/api"; // same site on Vercel
const $ = (s) => document.querySelector(s);
let cfg = null, results = [], scannedAt = 0;

// ---------- helpers ----------
const pct = (x, d = 2) => (x * 100).toFixed(d) + "%";
const usd = (x, d = 2) => "$" + Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtDate = (s) => new Date(s + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" });
const roundPrice = (p) => { const t = p < 3 ? 0.01 : 0.05; return Math.max(t, Math.round(p / t) * t).toFixed(2); };

function msg(el, text, kind = "err") {
  el.innerHTML = "";
  if (!text) return;
  const d = document.createElement("div");
  d.className = "msg " + kind;
  d.textContent = text;
  el.appendChild(d);
}

// Only placing an order needs the password; everything else is open.
async function api(path, opts = {}, password = "") {
  const headers = { "Content-Type": "application/json" };
  if (password) headers.Authorization = "Bearer " + password;
  const res = await fetch(API + path, { ...opts, headers });
  let body = null;
  try { body = await res.json(); } catch (e) {}
  if (!res.ok) {
    const d = body && body.detail;
    const text = typeof d === "string" ? d : d ? JSON.stringify(d) : `${res.status} ${res.statusText}`;
    const err = new Error(text); err.status = res.status; throw err;
  }
  return body;
}

function sparkline(vals) {
  if (!vals || vals.length < 2) return "";
  const w = 88, h = 24, min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const pts = vals.map((v, i) => `${(i / (vals.length - 1) * w).toFixed(1)},${(h - (v - min) / span * (h - 2) - 1).toFixed(1)}`).join(" ");
  const up = vals[vals.length - 1] >= vals[0];
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${up ? "var(--accent)" : "var(--risk)"}" stroke-width="1.5"/></svg>`;
}

function showApp() {
  const sel = $("#provider");
  sel.innerHTML = "";
  cfg.providers.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.name; o.textContent = p.name; o.dataset.model = p.default_model;
    sel.appendChild(o);
  });
  const hosted = [...sel.options].find((o) => !o.value.startsWith("Ollama"));
  if (hosted) sel.value = hosted.value;
  if (sel.options.length) {
    $("#model").value = sel.selectedOptions[0].dataset.model;
  } else {
    $("#analyze-btn").disabled = true;
    msg($("#ai-out"), "No AI key is set. Add GEMINI_API_KEY, ANTHROPIC_API_KEY or OMNIROUTE_BASE_URL in Vercel → Settings → Environment Variables.", "info");
  }
  $("#o-qty").max = cfg.max_contracts;
  $("#live-opt").classList.toggle("hidden", !cfg.live_allowed);
}
$("#provider").addEventListener("change", (e) => { $("#model").value = e.target.selectedOptions[0].dataset.model; });

// ---------- scan ----------
$("#scan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#scan-btn");
  btn.disabled = true; btn.textContent = "Scanning…";
  msg($("#scan-msg"), "");
  try {
    const data = await api("/scan", {
      method: "POST",
      body: JSON.stringify({
        tickers: $("#tickers").value.split(",").map((t) => t.trim()).filter(Boolean),
        delta: parseFloat($("#delta").value),
        min_dte: parseInt($("#min-dte").value, 10),
        max_dte: parseInt($("#max-dte").value, 10),
      }),
    });
    results = data.results; scannedAt = Date.now();
    if (!results.length) {
      $("#results").classList.add("hidden");
      const details = data.errors.length
        ? ` Reasons: ${data.errors.join(" · ")}`
        : "";
      msg($("#scan-msg"),
          `No results. Check the tickers or widen the days-out range.${details}`);
      return;
    }
    if (data.errors.length) msg($("#scan-msg"), data.errors.join(" · "), "info");
    renderResults();
  } catch (e) {
    msg($("#scan-msg"), e.message);
  } finally { btn.disabled = false; btn.textContent = "Scan"; }
});

function renderResults() {
  const top = results[0];
  $("#verdict-big").textContent = `${top.ticker} pays ${pct(top.yield)} this week`;
  $("#verdict-sub").textContent =
    `Sell the ${usd(top.strike, top.strike % 1 ? 2 : 0)} put expiring ${fmtDate(top.expiry)} for about ${usd(top.premium, 0)} per contract ` +
    `on ${usd(top.collateral, 0)} collateral. That is ${pct(top.otm, 1)} below the latest daily close (${top.quote_asof || "date unavailable"}), ${pct(top.annualized, 0)} annualized. Yahoo option quotes may be delayed; check the current broker quote before trading.`;

  const max = Math.max(...results.map((r) => r.yield));
  $("#bars").innerHTML = results.map((r, i) => `
    <div class="bar-row${i === 0 ? " win" : ""}">
      <span class="t">${r.ticker}</span>
      <div class="track"><div class="fill" style="width:0"></div></div>
      <span class="v">${pct(r.yield)}</span>
    </div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll("#bars .fill").forEach((f, i) => { f.style.width = (results[i].yield / max * 100) + "%"; }));

  $("#tbody").innerHTML = results.map((r) => `
    <tr>
      <td><strong>${r.ticker}</strong></td><td>${usd(r.price)}</td><td>${usd(r.strike)}</td>
      <td>${fmtDate(r.expiry)} <span class="muted">(${r.dte}d)</span></td><td>${r.delta.toFixed(2)}</td>
      <td>${usd(r.bid)}</td><td>${pct(r.iv, 0)}</td><td>${pct(r.otm, 1)}</td><td>${pct(r.annualized, 0)}</td>
      <td>${sparkline(r.trend)} <span class="muted">${(r.change_1m >= 0 ? "+" : "") + pct(r.change_1m, 1)}</span></td>
      <td><a class="trade" href="https://robinhood.com/options/chains/${encodeURIComponent(r.ticker)}" target="_blank" rel="noopener noreferrer"
        aria-label="Open ${r.ticker} options chain in Robinhood">Open in Robinhood ↗</a></td>
    </tr>`).join("");

  const sel = $("#o-ticker");
  sel.innerHTML = results.map((r) => `<option value="${r.ticker}">${r.ticker} ${usd(r.strike)} put</option>`).join("");
  if (!$("#analyze-btn").disabled) $("#ai-out").innerHTML = "";
  msg($("#order-msg"), "");
  selectTicket(top.ticker);
  $("#results").classList.remove("hidden");
  $("#message-bar").classList.remove("hidden");
  loadNews(results.map(r => r.ticker).join(","));
}

// ---------- AI ----------
$("#analyze-btn").addEventListener("click", async () => {
  const btn = $("#analyze-btn"), out = $("#ai-out");
  btn.disabled = true; btn.textContent = "Reviewing…";
  msg(out, "");
  try {
    const a = await api("/analyze", {
      method: "POST",
      body: JSON.stringify({ provider: $("#provider").value, model: $("#model").value, rows: results }),
    });
    out.innerHTML = "";
    const pick = document.createElement("p"); pick.className = "pick";
    pick.textContent = a.pick === "NONE" ? "No pick this week" : `Pick: ${a.pick}`;
    const conf = document.createElement("p"); conf.className = "muted"; conf.style.margin = "0";
    conf.textContent = `${a.confidence || "unknown"} confidence`;
    const sum = document.createElement("p"); sum.textContent = a.summary || "";
    out.append(pick, conf, sum);
    if (a.risks && a.risks.length) {
      const ul = document.createElement("ul"); ul.className = "risks";
      a.risks.forEach((r) => { const li = document.createElement("li"); li.textContent = r; ul.appendChild(li); });
      out.appendChild(ul);
    }
    if (a.per_ticker) {
      const n = document.createElement("div"); n.className = "notes";
      Object.entries(a.per_ticker).forEach(([t, note]) => { const p = document.createElement("div"); p.textContent = `${t}: ${note}`; n.appendChild(p); });
      out.appendChild(n);
    }
    if (results.some((r) => r.ticker === a.pick)) selectTicket(a.pick);
  } catch (e) { msg(out, e.message); }
  finally { btn.disabled = false; btn.textContent = "Review with AI"; }
});

// ---------- order ticket ----------
function current() { return results.find((r) => r.ticker === $("#o-ticker").value); }
function account() { return document.querySelector('input[name="acct"]:checked').value; }

function selectTicket(ticker) {
  $("#o-ticker").value = ticker;
  $("#o-limit").value = roundPrice(current().mid);
  $("#o-confirm").checked = false;
  updateTicket();
}

function updateTicket() {
  const r = current(); if (!r) return;
  const qty = Math.max(1, Math.min(parseInt($("#o-qty").value, 10) || 1, cfg.max_contracts));
  const limit = parseFloat($("#o-limit").value) || 0;
  const stale = Date.now() - scannedAt > cfg.stale_seconds * 1000;
  $("#o-summary").textContent =
    `Sell to open ${qty} × ${r.contract} (${r.ticker} ${usd(r.strike)} put, expires ${fmtDate(r.expiry)}) at ${usd(limit)} limit. ` +
    `Collects about ${usd(limit * 100 * qty, 0)} and ties up ${usd(r.strike * 100 * qty, 0)} in ${account()} account collateral.`;
  updateManual();
  $("#order-btn").disabled = stale || !$("#o-confirm").checked || limit <= 0 || !$("#o-pw").value;
  if (stale) msg($("#order-msg"), "Quotes are over 5 minutes old. Scan again before placing an order.", "info");
}

// Same order, written the way you'd enter it by hand at Robinhood or Fidelity
function manualFields() {
  const r = current(); if (!r) return [];
  const qty = Math.max(1, Math.min(parseInt($("#o-qty").value, 10) || 1, cfg.max_contracts));
  const limit = parseFloat($("#o-limit").value) || 0;
  const longDate = new Date(r.expiry + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  return [
    ["Ticker", r.ticker],
    ["Expiration", longDate],
    ["Strike", usd(r.strike)],
    ["Type", "Put"],
    ["Action", "Sell to open (Robinhood: Sell)"],
    ["Contracts", String(qty)],
    ["Order type", `Limit at ${usd(limit)} per share`],
    ["Time in force", "Day (good for day)"],
    ["Cash needed", usd(r.strike * 100 * qty, 0)],
  ];
}

function updateManual() {
  const dl = $("#m-fields"); dl.innerHTML = "";
  manualFields().forEach(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    dl.append(dt, dd);
  });
}

$("#m-copy").addEventListener("click", async () => {
  const text = manualFields().map(([k, v]) => `${k}: ${v}`).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    msg($("#m-msg"), "Copied. Paste it into a note next to your broker app.", "ok");
  } catch (e) {
    msg($("#m-msg"), "Couldn't copy automatically. Select the details above and copy them.", "info");
  }
});

["#o-ticker"].forEach((s) => $(s).addEventListener("change", () => selectTicket($("#o-ticker").value)));
["#o-qty", "#o-limit", "#o-pw", "#o-confirm"].forEach((s) => $(s).addEventListener("input", updateTicket));
document.querySelectorAll('input[name="acct"]').forEach((el) => el.addEventListener("change", () => { $("#o-confirm").checked = false; updateTicket(); }));
setInterval(() => { if (results.length) updateTicket(); }, 15000);

$("#order-btn").addEventListener("click", async () => {
  const r = current(), btn = $("#order-btn");
  const qty = parseInt($("#o-qty").value, 10);
  btn.disabled = true; btn.textContent = "Placing…";
  try {
    const o = await api("/order", {
      method: "POST",
      body: JSON.stringify({ contract: r.contract, qty, limit: parseFloat($("#o-limit").value), account: account(), ticket: r.ticket, manual_quote_confirmed: $("#o-confirm").checked }),
    }, $("#o-pw").value);
    msg($("#order-msg"), `Order placed in ${o.account}: ${o.id} (${o.status})`, "ok");
    $("#o-confirm").checked = false;
  } catch (e) { msg($("#order-msg"), e.status === 401 ? "Wrong order password." : e.message); }
  finally { btn.textContent = "Place order"; updateTicket(); }
});


// ---------- market news ----------
function timeAgo(ts) {
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts || 0)));
  if (!ts) return "Date unavailable";
  if (seconds < 60) return "Just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
async function loadNews(tickers) {
  const btn = $("#news-btn"), out = $("#news-out");
  btn.disabled = true; btn.textContent = "Loading…"; out.replaceChildren();
  try {
    const symbols = tickers || results.map(r => r.ticker).join(",") || "SPY,QQQ,NVDA";
    const data = await api("/news?tickers=" + encodeURIComponent(symbols));
    if (!data.items.length) {
      const p = document.createElement("p"); p.className = "news-empty";
      p.textContent = data.errors && data.errors.length ? "News provider unavailable for some symbols." : "No headlines found.";
      out.append(p); return;
    }
    const grid = document.createElement("div"); grid.className = "news-grid";
    for (const item of data.items) {
      let url;
      try { url = new URL(item.url); } catch (_) { continue; }
      if (url.protocol !== "https:") continue;
      const card = document.createElement("a"); card.className = "news-card";
      card.href = url.href; card.target = "_blank"; card.rel = "noopener noreferrer";
      const meta = document.createElement("div"); meta.className = "news-meta";
      const badge = document.createElement("span"); badge.className = "news-badge"; badge.textContent = item.ticker;
      const when = document.createElement("span"); when.textContent = timeAgo(item.published);
      meta.append(badge, when);
      const title = document.createElement("div"); title.className = "news-title"; title.textContent = item.title;
      const source = document.createElement("div"); source.className = "news-source"; source.textContent = item.source || "News source";
      card.append(meta, title, source); grid.append(card);
    }
    out.append(grid);
  } catch (error) {
    const p = document.createElement("p"); p.className = "news-empty"; p.textContent = "Could not load news: " + error.message; out.append(p);
  } finally { btn.disabled = false; btn.textContent = "Refresh"; }
}
$("#news-btn").addEventListener("click", () => loadNews());

async function sendScanResults(kind) {
  const btn = $(kind === "email" ? "#email-btn" : "#text-btn"), status = $("#message-status");
  btn.disabled = true; status.textContent = "Sending…";
  try {
    const response = await api(kind === "email" ? "/email-results" : "/text-results", {
      method: "POST", body: JSON.stringify({ results })
    }, $("#o-pw").value);
    status.textContent = `Sent to ${response.sent} configured recipient(s).`;
  } catch (error) { status.textContent = error.message; }
  finally { btn.disabled = false; }
}
$("#email-btn").addEventListener("click", () => sendScanResults("email"));
$("#text-btn").addEventListener("click", () => sendScanResults("text"));

// ---------- start ----------
const slow = setTimeout(() => msg($("#scan-msg"), "Starting up…", "info"), 2500);
api("/config")
  .then((c) => { cfg = c; showApp(); msg($("#scan-msg"), ""); })
  .catch((e) => msg($("#scan-msg"), "Couldn't reach the server: " + e.message))
  .finally(() => clearTimeout(slow));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
