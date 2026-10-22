"""AI review providers used by the options scanner API."""
import json
import os
import re

AI_SYSTEM = """Review cash-secured put candidates for an options seller.
Weigh premium against implied volatility, distance out of the money, one-month
trend, days to expiry, and the extra gap risk of leveraged ETFs. This is a
screening aid, not a promise of returns or personalized financial advice.
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
        model=model, contents=prompt,
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
        model=model, max_tokens=1000, system=AI_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)

PROVIDERS = {}
if os.getenv("GEMINI_API_KEY"):
    PROVIDERS["Gemini"] = (_ask_gemini, "gemini-3.8-flash")
if os.getenv("ANTHROPIC_API_KEY"):
    PROVIDERS["Claude"] = (_ask_claude, "claude-haiku-4-5")

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
