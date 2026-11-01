"""Ticker follow / unfollow endpoints for insider-alert subscriptions."""
import logging
import os

from flask import Blueprint, jsonify, request
from pymongo import MongoClient, ASCENDING

log = logging.getLogger(__name__)

router = Blueprint("follow", __name__, url_prefix="/api/signal")

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


@router.post("/follow")
def follow():
    """Subscribe an email address to insider-filing alerts for a ticker."""
    err = _require_key()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    ticker = str(body.get("ticker", "")).strip().upper()
    if not email or not ticker:
        return jsonify({"detail": "email and ticker are required"}), 400

    col = _db()["follows"]
    col.create_index([("email", ASCENDING), ("ticker", ASCENDING)], unique=True)
    col.update_one(
        {"email": email, "ticker": ticker},
        {"$setOnInsert": {"email": email, "ticker": ticker, "last_seen": None}},
        upsert=True,
    )
    return jsonify({"ok": True, "email": email, "ticker": ticker})


@router.delete("/follow")
def unfollow():
    """Unsubscribe an email address from alerts for a ticker."""
    err = _require_key()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    ticker = str(body.get("ticker", "")).strip().upper()
    if not email or not ticker:
        return jsonify({"detail": "email and ticker are required"}), 400

    _db()["follows"].delete_one({"email": email, "ticker": ticker})
    return jsonify({"ok": True})


@router.get("/follows/<path:email>")
def list_follows(email: str):
    """Return all tickers followed by a given email address."""
    err = _require_key()
    if err:
        return err
    email = email.strip().lower()
    tickers = [
        doc["ticker"]
        for doc in _db()["follows"].find({"email": email}, {"_id": 0, "ticker": 1})
    ]
    return jsonify({"email": email, "tickers": tickers})
