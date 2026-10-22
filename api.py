"""
Put scanner API for Render, called by the Vercel frontend.

Render start command: uvicorn api:app --host 0.0.0.0 --port $PORT
Environment:
  ALLOWED_ORIGINS=https://frontend-option-trading-nu.vercel.app
  APP_PASSWORD (required to submit orders)
  GEMINI_API_KEY and/or ANTHROPIC_API_KEY
  BROKER=alpaca (default) or robinhood
  ALPACA_API_KEY, ALPACA_SECRET_KEY, or RH_USERNAME, RH_PASSWORD
  MAX_CONTRACTS=1, MAX_COLLATERAL_PCT=0.25, LIVE_TRADING=false
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

MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "1"))
MAX_COLLATERAL_PCT = float(os.getenv("MAX_COLLATERAL_PCT", "0.25"))
LIVE_ALLOWED = os.getenv("LIVE_TRADING", "").lower() == "true"
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
STALE_SECONDS = 300

app = FastAPI(title="Put scanner API")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "https://frontend-option-trading-nu.vercel.app"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Contract symbol -> {strike, scan timestamp}. Kept in memory for recent scan orders.
_recent = {}


def public_access():
    # Scans and AI review are intentionally public. Orders use require_order_password.
    return None


def require_order_password(authorization: str = Header(default="")):
    if not APP_PASSWORD:
        raise HTTPException(503, "Order password is not configured on the server.")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token.encode(), APP_PASSWORD.encode()):
        raise HTTPException(401, "Wrong order password.")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/config", dependencies=[Depends(public_access)])
def config():
    return {
        "providers": [
            {"name": name, "default_model": model}
            for name, (_, model) in ai.PROVIDERS.items()
        ],
        "max_contracts": MAX_CONTRACTS,
        "max_collateral_pct": MAX_COLLATERAL_PCT,
        "live_allowed": LIVE_ALLOWED and bool(APP_PASSWORD),
        "stale_seconds": STALE_SECONDS,
    }


class ScanRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=12)
    delta: float = Field(0.15, ge=0.05, le=0.40)
    min_dte: int = Field(4, ge=0, le=60)
    max_dte: int = Field(10, ge=1, le=60)
    rate: float = Field(0.04, ge=0, le=0.15)


@app.post("/scan", dependencies=[Depends(public_access)])
def scan(req: ScanRequest):
    results, errors = [], []
    for raw in req.tickers:
        ticker = raw.strip().upper()
        if not ticker:
            continue
        try:
            result = scanner.scan_ticker(
                ticker, req.delta, req.min_dte, req.max_dte, req.rate
            )
            results.append(result)
            _recent[result["contract"]] = {
                "strike": result["strike"],
                "at": time.time(),
            }
        except Exception as exc:
            errors.append(
                str(exc) if isinstance(exc, ValueError) else f"{ticker}: {exc}"
            )
    results.sort(key=lambda row: row["yield"], reverse=True)
    return {"results": results, "errors": errors, "scanned_at": time.time()}


class AnalyzeRequest(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=100)
    rows: list[dict] = Field(min_length=1, max_length=12)


@app.post("/analyze", dependencies=[Depends(public_access)])
def analyze(req: AnalyzeRequest):
    if req.provider not in ai.PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {req.provider}")
    rows = [
        {key: value for key, value in row.items() if key not in ("contract", "trend")}
        for row in req.rows
    ]
    try:
        return ai.analyze(req.provider, req.model, rows)
    except Exception as exc:
        raise HTTPException(502, f"AI request failed: {exc}") from exc


class OrderRequest(BaseModel):
    contract: str = Field(min_length=10, max_length=30)
    qty: int = Field(ge=1)
    limit: float = Field(gt=0)
    account: Literal["paper", "live"] = "paper"


@app.post("/order", dependencies=[Depends(require_order_password)])
def order(req: OrderRequest):
    if req.account == "live" and not LIVE_ALLOWED:
        raise HTTPException(403, "Live trading is turned off on the server.")
    if req.qty > MAX_CONTRACTS:
        raise HTTPException(400, f"Max {MAX_CONTRACTS} contract(s) per order.")
    info = _recent.get(req.contract)
    if not info:
        raise HTTPException(400, "Contract isn't from a recent scan. Scan again.")
    if time.time() - info["at"] > STALE_SECONDS:
        raise HTTPException(409, "Scan is over 5 minutes old. Scan again.")

    collateral = info["strike"] * 100 * req.qty
    try:
        trading_client = broker.client(paper=(req.account == "paper"))
        buying_power = broker.buying_power(trading_client)
    except KeyError:
        raise HTTPException(503, "Broker API keys are not set on the server.")
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach broker: {exc}") from exc
    if collateral > buying_power * MAX_COLLATERAL_PCT:
        raise HTTPException(
            400,
            f"Blocked: ${collateral:,.0f} collateral is over "
            f"{MAX_COLLATERAL_PCT:.0%} of ${buying_power:,.0f} buying power.",
        )
    try:
        placed_order = broker.sell_put(
            trading_client, req.contract, req.qty, req.limit
        )
    except Exception as exc:
        raise HTTPException(502, f"Order rejected: {exc}") from exc
    return {
        "id": str(placed_order.id),
        "status": str(placed_order.status),
        "account": req.account,
    }
