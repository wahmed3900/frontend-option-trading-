"""
Weekly put scanner — the whole app in one file (page + API), Flask on Vercel.

Vercel: set Application Preset to "Flask". Add these in
Settings → Environment Variables, then redeploy:
  APP_PASSWORD                      (required)
  GEMINI_API_KEY and/or ANTHROPIC_API_KEY
  ALPACA_API_KEY, ALPACA_SECRET_KEY (paper keys to start)
  MAX_CONTRACTS=1, MAX_COLLATERAL_PCT=0.25, LIVE_TRADING=false

On your Mac:  APP_PASSWORD=test python app.py  → http://localhost:5000
"""
import hashlib
import hmac
import json
import math
import os
import re
import time
from datetime import date, datetime
from functools import wraps

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "1"))
MAX_COLLATERAL_PCT = float(os.getenv("MAX_COLLATERAL_PCT", "0.25"))
LIVE_ALLOWED = os.getenv("LIVE_TRADING", "").lower() == "true"
STALE_SECONDS = 300


# ---------------- helpers ----------------
def fail(status, message):
    return jsonify({"detail": message}), status


def require_password(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not APP_PASSWORD:
            return fail(503, "APP_PASSWORD is not set in Vercel.")
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
        "trend": closes, "ticket": {"ts": ts, "sig": sign(contract, ts)},
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


def providers():
    p = {}
    if os.getenv("GEMINI_API_KEY"):
        p["Gemini"] = (ask_gemini, "gemini-2.5-flash")
    if os.getenv("ANTHROPIC_API_KEY"):
        p["Claude"] = (ask_claude, "claude-haiku-4-5")
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


@app.get("/api/config")
@require_password
def config():
    return jsonify({
        "providers": [{"name": n, "default_model": m} for n, (_, m) in providers().items()],
        "max_contracts": MAX_CONTRACTS,
        "max_collateral_pct": MAX_COLLATERAL_PCT,
        "live_allowed": LIVE_ALLOWED,
        "stale_seconds": STALE_SECONDS,
    })


@app.post("/api/scan")
@require_password
def scan():
    body = request.get_json(silent=True) or {}
    tickers = [str(t).strip().upper() for t in body.get("tickers", []) if str(t).strip()][:12]
    if not tickers:
        return fail(400, "Enter at least one ticker.")
    delta = min(max(num(body.get("delta"), 0.20), 0.05), 0.40)
    min_dte = int(min(max(num(body.get("min_dte"), 4), 0), 60))
    max_dte = int(min(max(num(body.get("max_dte"), 10), 1), 60))

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
@require_password
def analyze():
    body = request.get_json(silent=True) or {}
    p = providers()
    name, model = body.get("provider"), str(body.get("model") or "")[:100]
    if name not in p:
        return fail(400, "That AI provider has no key set in Vercel.")
    rows = [{k: v for k, v in r.items() if k not in ("contract", "trend", "ticket")}
            for r in (body.get("rows") or [])[:12] if isinstance(r, dict)]
    if not rows:
        return fail(400, "Scan first.")
    fn, default_model = p[name]
    try:
        result = parse_json(fn("Candidates (yield = bid / strike):\n" + json.dumps(rows, indent=2), model or default_model))
    except Exception as e:
        return fail(502, f"AI request failed: {e}")
    if result.get("pick") not in {r["ticker"] for r in rows}:
        result["pick"] = "NONE"
    return jsonify(result)


@app.post("/api/order")
@require_password
def order():
    body = request.get_json(silent=True) or {}
    contract = str(body.get("contract", ""))
    qty = int(num(body.get("qty"), 0))
    limit = num(body.get("limit"), 0)
    account = body.get("account", "paper")
    ticket = body.get("ticket") or {}

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
    if time.time() - ts > STALE_SECONDS:
        return fail(409, "Quotes are over 5 minutes old. Scan again.")

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
        o = tc.submit_order(LimitOrderRequest(
            symbol=contract, qty=qty, side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
            time_in_force=TimeInForce.DAY, limit_price=round_price(limit),
        ))
    except Exception as e:
        return fail(502, f"Order rejected: {e}")
    return jsonify({"id": str(o.id), "status": str(o.status), "account": account})


# ---------------- page ----------------
@app.get("/")
def home():
    return Response(PAGE_HTML, mimetype="text/html")


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

/* login */
.login { max-width: 360px; margin: 18vh auto 0; }
.login p { color: var(--muted); }
.login form { display: grid; gap: 12px; }

footer { margin-top: 48px; font-size: .85rem; color: var(--muted); max-width: 70ch; }
</style>
</head>
<body>

<section id="login" class="login wrap hidden">
  <h1>Weekly put scanner</h1>
  <form id="login-form">
    <div>
      <label for="pw">Password</label>
      <input id="pw" type="password" autocomplete="current-password" required>
    </div>
    <button type="submit" id="login-btn">Sign in</button>
  </form>
  <div id="login-msg"></div>
</section>

<main id="app" class="wrap hidden">
  <h1>Weekly put scanner</h1>

  <form id="scan-form" class="settings">
    <div class="tick"><label for="tickers">Tickers</label><input id="tickers" value="NVDA, META, NVDL, SOXL"></div>
    <div><label for="delta">Target delta</label><input id="delta" type="number" min="0.05" max="0.40" step="0.01" value="0.20"></div>
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
          <th>IV</th><th>Below price</th><th>Annualized</th><th>1 month</th>
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
        <label class="check"><input type="checkbox" id="o-confirm"> I've checked this against my broker and want to place it.</label>
        <button id="order-btn" type="button" disabled>Place order</button>
        <div id="order-msg"></div>
      </div>
    </div>
  </section>

  <footer>
    Data from Yahoo Finance, delayed about 15 minutes. Delta is calculated from Yahoo's implied volatility.
    The AI only reviews; strike, size, and price come from the scan and your inputs.
    Leveraged ETFs pay more because they can gap through the strike. Not financial advice.
    <br><button class="secondary" id="logout" type="button" style="margin-top:12px">Sign out</button>
  </footer>
</main>

<script>
const API = "/api"; // same site on Vercel
const $ = (s) => document.querySelector(s);
let token = "";
try { token = sessionStorage.getItem("ps_pw") || ""; } catch (e) {}
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

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    ...opts,
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
  });
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

// ---------- login ----------
async function tryLogin(pw) {
  token = pw;
  const slow = setTimeout(() => msg($("#login-msg"), "Starting up…", "info"), 2500);
  try {
    cfg = await api("/config");
    try { sessionStorage.setItem("ps_pw", pw); } catch (e) {}
    showApp();
  } catch (e) {
    try { sessionStorage.removeItem("ps_pw"); } catch (x) {}
    $("#login").classList.remove("hidden");
    $("#app").classList.add("hidden");
    msg($("#login-msg"), e.status === 401 ? "Wrong password." : "Couldn't reach the server: " + e.message);
  } finally { clearTimeout(slow); }
}

$("#login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("#login-btn").disabled = true;
  await tryLogin($("#pw").value);
  $("#login-btn").disabled = false;
});

$("#logout").addEventListener("click", () => {
  try { sessionStorage.removeItem("ps_pw"); } catch (e) {}
  location.reload();
});

function showApp() {
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
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
    msg($("#ai-out"), "No AI key is set. Add GEMINI_API_KEY or ANTHROPIC_API_KEY in Vercel → Settings → Environment Variables.", "info");
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
    if (data.errors.length) msg($("#scan-msg"), data.errors.join(" · "), "info");
    if (!results.length) { $("#results").classList.add("hidden"); msg($("#scan-msg"), "No results. Check the tickers or widen the days-out range."); return; }
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
    `on ${usd(top.collateral, 0)} collateral. That's ${pct(top.otm, 1)} below today's price, ${pct(top.annualized, 0)} annualized.`;

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
    </tr>`).join("");

  const sel = $("#o-ticker");
  sel.innerHTML = results.map((r) => `<option value="${r.ticker}">${r.ticker} ${usd(r.strike)} put</option>`).join("");
  if (!$("#analyze-btn").disabled) $("#ai-out").innerHTML = "";
  msg($("#order-msg"), "");
  selectTicket(top.ticker);
  $("#results").classList.remove("hidden");
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
  $("#order-btn").disabled = stale || !$("#o-confirm").checked || limit <= 0;
  if (stale) msg($("#order-msg"), "Quotes are over 5 minutes old. Scan again before placing an order.", "info");
}

["#o-ticker"].forEach((s) => $(s).addEventListener("change", () => selectTicket($("#o-ticker").value)));
["#o-qty", "#o-limit", "#o-confirm"].forEach((s) => $(s).addEventListener("input", updateTicket));
document.querySelectorAll('input[name="acct"]').forEach((el) => el.addEventListener("change", () => { $("#o-confirm").checked = false; updateTicket(); }));
setInterval(() => { if (results.length) updateTicket(); }, 15000);

$("#order-btn").addEventListener("click", async () => {
  const r = current(), btn = $("#order-btn");
  const qty = parseInt($("#o-qty").value, 10);
  btn.disabled = true; btn.textContent = "Placing…";
  try {
    const o = await api("/order", {
      method: "POST",
      body: JSON.stringify({ contract: r.contract, qty, limit: parseFloat($("#o-limit").value), account: account(), ticket: r.ticket }),
    });
    msg($("#order-msg"), `Order placed in ${o.account}: ${o.id} (${o.status})`, "ok");
    $("#o-confirm").checked = false;
  } catch (e) { msg($("#order-msg"), e.message); }
  finally { btn.textContent = "Place order"; updateTicket(); }
});

// ---------- start ----------
if (token) tryLogin(token); else $("#login").classList.remove("hidden");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
