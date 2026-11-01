import os
import secrets
from fastapi import APIRouter, Depends, Header, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from smart_signal import get_cik

router = APIRouter(prefix="/api/signal", tags=["signal"])

_client = AsyncIOMotorClient(os.environ["MONGO_URI"])
follows = _client["stockai"]["signal_follows"]


def get_user(
    x_user_email: str = Header(...),
    x_api_key: str = Header(...),
) -> str:
    """Only your Next.js server knows SIGNAL_API_KEY, so users can't fake emails."""
    expected = os.environ.get("SIGNAL_API_KEY", "")
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Unauthorized")
    return x_user_email.strip().lower()


class FollowIn(BaseModel):
    ticker: str


@router.post("/follow")
async def follow(body: FollowIn, email: str = Depends(get_user)):
    """Start following a ticker."""
    ticker = body.ticker.strip().upper()
    cik = await get_cik(ticker)  # 404s if the ticker isn't an SEC filer
    await follows.update_one(
        {"email": email, "ticker": ticker},
        {"$setOnInsert": {
            "email": email,
            "ticker": ticker,
            "cik": cik,
            "last_seen": None,  # newest filing already alerted (used in step 3)
        }},
        upsert=True,
    )
    return {"following": ticker}


@router.delete("/follow/{ticker}")
async def unfollow(ticker: str, email: str = Depends(get_user)):
    """Stop following a ticker."""
    await follows.delete_one({"email": email, "ticker": ticker.strip().upper()})
    return {"unfollowed": ticker.upper()}


@router.get("/following")
async def following(email: str = Depends(get_user)):
    """List the tickers this user follows."""
    docs = await follows.find({"email": email}, {"_id": 0, "ticker": 1}).to_list(100)
    return {"tickers": [d["ticker"] for d in docs]}
