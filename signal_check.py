"""Periodic checker: finds new SEC Form 4 filings for followed tickers and queues alerts."""
import logging
import os
from datetime import date

import requests
from flask import Blueprint, jsonify, request
from pymongo import MongoClient

from smart_signal import _get_cik, SEC_HEADERS
from signal_email import send_alerts

log = logging.getLogger(__name__)

router = Blueprint("check", __name__, url_prefix="/api/signal")

_client: MongoClient | None = None


def _db():
    global _client
    if _client is None:
        uri = os.environ.get("MONGO_URI", "")
        if not uri:
            raise RuntimeError("MONGO_URI is not set")
        _client = MongoClient(uri)
    return _client["stockai"]


def _require_key():
    expected = os.environ.get("SIGNAL_API_KEY", "")
    if not expected:
        return jsonify({"detail": "SIGNAL_API_KEY not configured on the server"}), 500
    provided = request.headers.get("X-Signal-Key", "")
    if not provided or provided != expected:
        return jsonify({"detail": "Unauthorized"}), 401
    return None


def _filings_since(ticker: str, since: str | None) -> list[dict]:
    """Return Form 4 filings for ticker filed after since (YYYY-MM-DD)."""
    cik = _get_cik(ticker)
    if not cik:
        return []
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
    except Exception:
        log.warning("SEC fetch failed for %s", ticker)
        return []

    results = []
    for form, filing_date, accession, document in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form != "4":
            continue
        if since and filing_date <= since:
            break
        results.append({
            "ticker": ticker,
            "date": filing_date,
            "url": (
                "https://www.sec.gov/Archives/edgar/data/"
                + str(int(cik)) + "/" + accession.replace("-", "") + "/" + document
            ),
        })
    return results


@router.post("/check")
def run_check():
    """Called by Cloud Scheduler every 15 min."""
    err = _require_key()
    if err:
        return err

    col = _db()["follows"]
    today = date.today().isoformat()

    all_docs = list(col.find({}, {"_id": 0, "email": 1, "ticker": 1, "last_seen": 1}))
    unique_tickers = {}
    for doc in all_docs:
        t = doc["ticker"]
        if t not in unique_tickers:
            unique_tickers[t] = doc.get("last_seen")

    ticker_filings = {}
    for ticker, last_seen in unique_tickers.items():
        ticker_filings[ticker] = _filings_since(ticker, last_seen)

    for ticker, filings in ticker_filings.items():
        if filings:
            col.update_many({"ticker": ticker}, {"$set": {"last_seen": filings[0]["date"]}})
        elif unique_tickers[ticker] is None:
            col.update_many({"ticker": ticker, "last_seen": None}, {"$set": {"last_seen": today}})

    user_items = {}
    for doc in all_docs:
        if doc.get("last_seen") is None:
            continue
        filings = ticker_filings.get(doc["ticker"], [])
        if filings:
            user_items.setdefault(doc["email"], []).extend(filings)

    for email, items in user_items.items():
        send_alerts(email, items)

    return jsonify({"ok": True, "alerts_sent": len(user_items)})
