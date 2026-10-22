"""AI review providers used by the options scanner API."""
import json
import os
import re

AI_SYSTEM = """Review cash-secured put candidates for an options seller.
Weigh the bid/strike premium against implied volatility, distance out of the
money, one-month trend, days to expiry, and the extra gap risk of leveraged ETFs.
This is a screening aid, not a promise of returns or personalized financial advice.
Respond only with JSON in this shape:
{"pick":"<TICKER or NONE>","confidence":"low|medium|high",
 "summary":"<2-3 sentences>","risks":["..."],
 "per_ticker":{"<TICKER>":"<one line>"}}
Only pick a ticker present in the supplied data."""


def _ask_gemini(prompt, model):
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=AI_SYSTEM,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def _ask_claude(prompt, model):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=AI_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_omniroute(prompt, model):
    import requests

    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if not base_url:
        raise ValueError("OMNIROUTE_BASE_URL is not configured.")
    if not model:
        raise ValueError("Set OMNIROUTE_MODEL or enter a model name.")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    headers = {}
    api_key = os.getenv("OMNIROUTE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        timeout=60,
        json={
            "model": model,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": AI_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        },
    )
    if not response.ok:
        raise ValueError(f"OmniRoute returned HTTP {response.status_code}.")
    return response.json()["choices"][0]["message"]["content"] or ""


PROVIDERS = {}
if os.getenv("GEMINI_API_KEY"):
    PROVIDERS["Gemini"] = (_ask_gemini, "gemini-3.8-flash")
if os.getenv("ANTHROPIC_API_KEY"):
    PROVIDERS["Claude"] = (_ask_claude, "claude-haiku-4-5")
if os.getenv("OMNIROUTE_BASE_URL"):
    PROVIDERS["OmniRoute"] = (_ask_omniroute, os.getenv("OMNIROUTE_MODEL", ""))


def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)


def analyze(provider, model, rows):
    if provider not in PROVIDERS:
        raise ValueError("That AI provider has no key configured.")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Scan first.")
    ask, default_model = PROVIDERS[provider]
    prompt = (
        "Candidates (yield = bid / strike):\n"
        + json.dumps(rows[:12], ensure_ascii=False, allow_nan=False)
    )
    result = _parse_json(ask(prompt, model or default_model))
    if not isinstance(result, dict):
        raise ValueError("The AI provider returned an invalid response.")
    tickers = {str(row.get("ticker", "")) for row in rows if isinstance(row, dict)}
    pick = result.get("pick")
    if not isinstance(pick, str) or pick not in tickers:
        pick = "NONE"
    confidence = result.get("confidence")
    if confidence not in {"low", "medium", "high"}:
        confidence = "unknown"
    risks = result.get("risks")
    if not isinstance(risks, list):
        risks = []
    per_ticker = result.get("per_ticker")
    if not isinstance(per_ticker, dict):
        per_ticker = {}
    return {
        "pick": pick,
        "confidence": confidence,
        "summary": str(result.get("summary", ""))[:2000],
        "risks": [str(item)[:500] for item in risks[:20]],
        "per_ticker": {
            str(ticker)[:20]: str(note)[:500]
            for ticker, note in list(per_ticker.items())[:12]
        },
    }
