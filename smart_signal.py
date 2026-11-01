"""SEC insider filing routes for the weekly options scanner."""
import re
import threading
import time

import requests
from flask import Blueprint, jsonify, request


router = Blueprint("signal", __name__, url_prefix="/api/signal")
SEC_HEADERS = {"User-Agent": "StockAI wahmed07860786@gmail.com"}
_CIK_TTL_SECONDS = 86_400
_cik_cache: dict[str, str] = {}
_cik_loaded_at = 0.0
_cik_lock = threading.Lock()


def _get_cik(ticker: str) -> str | None:
    global _cik_loaded_at
    with _cik_lock:
        if not _cik_cache or time.time() - _cik_loaded_at > _CIK_TTL_SECONDS:
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=SEC_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            rows = response.json().values()
            refreshed = {
                str(row["ticker"]).upper(): str(row["cik_str"]).zfill(10)
                for row in rows
            }
            _cik_cache.clear()
            _cik_cache.update(refreshed)
            _cik_loaded_at = time.time()
        return _cik_cache.get(ticker.upper())


@router.get("/insider/<ticker>")
def insider_filings(ticker: str):
    """Return recent SEC Form 4 filing dates and EDGAR URLs for a ticker."""
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.-]{1,10}", ticker):
        return jsonify({"detail": "Invalid ticker"}), 400

    try:
        limit = min(100, max(1, int(request.args.get("limit", "20"))))
    except (TypeError, ValueError):
        limit = 20

    try:
        cik = _get_cik(ticker)
        if not cik:
            return jsonify({"detail": f"No SEC filer found for {ticker}"}), 404

        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=SEC_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        recent = response.json().get("filings", {}).get("recent", {})
        filings = []
        for form, date, accession, document in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            recent.get("primaryDocument", []),
        ):
            if form != "4":
                continue
            filings.append({
                "date": date,
                "url": (
                    "https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession.replace('-', '')}/{document}"
                ),
            })
            if len(filings) >= limit:
                break
        response = jsonify({"ticker": ticker, "filings": filings})
        response.headers["Cache-Control"] = "public, max-age=900, stale-while-revalidate=3600"
        return response
    except requests.RequestException:
        return jsonify({"detail": "SEC filings are temporarily unavailable. Please try again."}), 502
    except (KeyError, TypeError, ValueError):
        return jsonify({"detail": "SEC returned filing data in an unexpected format."}), 502
