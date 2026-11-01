"""Ticker follow / unfollow endpoints for insider-alert subscriptions."""import loggingimport osfrom flask import Blueprint, jsonify, requestfrom pymongo import MongoClient, ASCENDINGlog = logging.getLogger(__name__)router = Blueprint("follow", __name__, url_prefix="/api/signal")_client: MongoClient | None = Nonedef _db():    global _client    if _client is None:        uri = os.environ.get("MONGO_URI", "")        if not uri:            raise RuntimeError("MONGO_URI is not set")        _client = MongoClient(uri)    return _client["stockai"]def _require_key():    expected = os.environ.get("SIGNAL_API_KEY", "")    if not expected:        return jsonify({"detail": "SIGNAL_API_KEY not configured on the server"}), 500    provided = request.headers.get("X-Signal-Key", "")    if not provided or provided != expected:        return jsonify({"detail": "Unauthorized"}), 401    return None@router.post("/follow")def follow():    """Subscribe an email address to insider-filing alerts for a ticker."""    err = _require_key()    if err:        return err    body = request.get_json(silent=True) or {}    email = str(body.get("email", "")).strip().lower()    ticker = str(body.get("ticker", "")).strip().upper()    if not email or not ticker:        return jsonify({"detail": "email and ticker are required"}), 400    col = _db()["follows"]    col.create_index([("email", ASCENDING), ("ticker", ASCENDING)], unique=True)    col.update_one(        {"email": email, "ticker": ticker},        {"$setOnInsert": {"email": email, "ticker": ticker, "last_seen": None}},        upsert=True,    )    return jsonify({"ok": True, "email": email, "ticker": ticker})@router.delete("/follow")def unfollow():    """Unsubscribe an email address from alerts for a ticker."""    err = _require_key()    if err:        return err

# ---------------------------------------------------------------------------
# App-password-protected browser routes for end-user subscriptions
# ---------------------------------------------------------------------------
import re as _re

_EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _check_app_password():
    app_pw = os.environ.get("APP_PASSWORD", "")
    if app_pw and request.headers.get("X-App-Password", "") != app_pw:
        return jsonify({"detail": "Unauthorized"}), 401
    return None


@router.post("/subscribe")
def subscribe():
    """Subscribe an email to insider-filing alerts (requires X-App-Password header)."""
    err = _check_app_password()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    ticker = str(body.get("ticker", "")).strip().upper()
    if not email or not _EMAIL_RE.match(email) or not ticker:
        return jsonify({"detail": "valid email and ticker are required"}), 400
    col = _db()["follows"]
    col.create_index([("email", ASCENDING), ("ticker", ASCENDING)], unique=True)
    col.update_one(
        {"email": email, "ticker": ticker},
        {"$setOnInsert": {"email": email, "ticker": ticker, "last_seen": None}},
        upsert=True,
    )
    return jsonify({"ok": True, "email": email, "ticker": ticker})


@router.delete("/subscribe")
def unsubscribe():
    """Unsubscribe an email from alerts (requires X-App-Password header)."""
    err = _check_app_password()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    ticker = str(body.get("ticker", "")).strip().upper()
    if not email or not ticker:
        return jsonify({"detail": "email and ticker are required"}), 400
    _db()["follows"].delete_one({"email": email, "ticker": ticker})
    return jsonify({"ok": True})


@router.get("/subscriptions/<path:email>")
def list_subscriptions(email: str):
    """Return followed tickers for an email (requires X-App-Password header)."""
    err = _check_app_password()
    if err:
        return err
    email = email.strip().lower()
    tickers = [
        doc["ticker"]
        for doc in _db()["follows"].find({"email": email}, {"_id": 0, "ticker": 1})
    ]
    return jsonify({"email": email, "tickers": tickers})
