"""
Put scanner API — runs on Render, called by the Vercel frontend.

Start:  uvicorn api:app --host 0.0.0.0 --port $PORT
Env:    APP_PASSWORD (required), ALLOWED_ORIGINS (your Vercel URL),
        GEMINI_API_KEY / ANTHROPIC_API_KEY, ALPACA_API_KEY / ALPACA_SECRET_KEY,
        MAX_CONTRACTS, MAX_COLLATERAL_PCT, LIVE_TRADING
"""
import hmac
import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import ai
import broker
import scanner

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "1"))
MAX_COLLATERAL_PCT = float(os.getenv("MAX_COLLATERAL_PCT", "0.25"))
LIVE_ALLOWED = os.getenv("LIVE_TRADING", "").lower() == "true"
STALE_SECONDS = 300

app = FastAPI(title="Put scanner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# contract symbol -> {"strike", "at"}; only contracts from a recent scan can be ordered
_recent = {}


def auth(authorization: str = Header(default="")):
    if not APP_PASSWORD:
        raise HTTPException(503, "APP_PASSWORD is not set on the server.")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token.encode(), APP_PASSWORD.encode()):
        raise HTTPException(401, "Wrong password.")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/config", dependencies=[Depends(auth)])
def config():
    return {
        "providers": [{"name": n, "default_model": m} for n, (_, m) in ai.PROVIDERS.items()],
        "max_contracts": MAX_CONTRACTS,
        "max_collateral_pct": MAX_COLLATERAL_PCT,
        "live_allowed": LIVE_ALLOWED,
        "stale_seconds": STALE_SECONDS,
    }


class ScanRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=12)
    delta: float = Field(0.20, ge=0.05, le=0.40)
    min_dte: int = Field(4, ge=0, le=60)
    max_dte: int = Field(10, ge=1, le=60)
    rate: float = Field(0.04, ge=0, le=0.15)


@app.post("/scan", dependencies=[Depends(auth)])
def scan(req: ScanRequest):
    results, errors = [], []
    for raw in req.tickers:
        t = raw.strip().upper()
        if not t:
            continue
        try:
            r = scanner.scan_ticker(t, req.delta, req.min_dte, req.max_dte, req.rate)
            results.append(r)
            _recent[r["contract"]] = {"strike": r["strike"], "at": time.time()}
        except Exception as e:
            errors.append(str(e) if isinstance(e, ValueError) else f"{t}: {e}")
    results.sort(key=lambda r: r["yield"], reverse=True)
    return {"results": results, "errors": errors, "scanned_at": time.time()}


class AnalyzeRequest(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=100)
    rows: list[dict] = Field(min_length=1, max_length=12)


@app.post("/analyze", dependencies=[Depends(auth)])
def analyze(req: AnalyzeRequest):
    if req.provider not in ai.PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {req.provider}")
    rows = [{k: v for k, v in r.items() if k not in ("contract", "trend")} for r in req.rows]
    try:
        return ai.analyze(req.provider, req.model, rows)
    except Exception as e:
        raise HTTPException(502, f"AI request failed: {e}")


class OrderRequest(BaseModel):
    contract: str = Field(min_length=10, max_length=30)
    qty: int = Field(ge=1)
    limit: float = Field(gt=0)
    account: Literal["paper", "live"] = "paper"


@app.post("/order", dependencies=[Depends(auth)])
def order(req: OrderRequest):
    if req.account == "live" and not LIVE_ALLOWED:
        raise HTTPException(403, "Live trading is turned off on the server.")
    if req.qty > MAX_CONTRACTS:
        raise HTTPException(400, f"Max {MAX_CONTRACTS} contract(s) per order.")
    info = _recent.get(req.contract)
    if not info:
        raise HTTPException(400, "Contract isn't from a recent scan. Scan again.")
    if time.time() - info["at"] > STALE_SECONDS:
        raise HTTPException(409, "Quotes are over 5 minutes old. Scan again.")

    collateral = info["strike"] * 100 * req.qty
    try:
        tc = broker.client(paper=(req.account == "paper"))
        bp = broker.buying_power(tc)
    except KeyError:
        raise HTTPException(503, "Alpaca keys are not set on the server.")
    except Exception as e:
        raise HTTPException(502, f"Couldn't reach Alpaca: {e}")
    if collateral > bp * MAX_COLLATERAL_PCT:
        raise HTTPException(
            400,
            f"Blocked: ${collateral:,.0f} collateral is over {MAX_COLLATERAL_PCT:.0%} "
            f"of ${bp:,.0f} buying power.",
        )
    try:
        o = broker.sell_put(tc, req.contract, req.qty, req.limit)
    except Exception as e:
        raise HTTPException(502, f"Order rejected: {e}")
    return {"id": str(o.id), "status": str(o.status), "account": req.account}
