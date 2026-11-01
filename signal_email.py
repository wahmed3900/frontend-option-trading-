"""Resend-based email alerts for new SEC insider filings."""
import logging
import os

import resend

log = logging.getLogger(__name__)

_ALERT_FROM = os.environ.get("ALERT_FROM", "StockAI <onboarding@resend.dev>")
_APP_URL = os.environ.get("APP_URL", "https://frontend-option-trading-nu.vercel.app")


def send_alerts(email: str, items: list[dict]) -> None:
    """Send one digest email listing all new insider filings for this user."""
    if not items:
        return

    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        log.error("RESEND_API_KEY not set - skipping alert for %s", email)
        return

    resend.api_key = api_key
    tickers_seen = sorted({i["ticker"] for i in items})

    rows = "".join(
        "<tr>"
        "<td style='padding:8px 12px;border-bottom:1px solid #eee;font-weight:600'>" + item["ticker"] + "</td>"
        "<td style='padding:8px 12px;border-bottom:1px solid #eee'>" + item["date"] + "</td>"
        "<td style='padding:8px 12px;border-bottom:1px solid #eee'>"
        "<a href='" + item["url"] + "' style='color:#4f46e5;text-decoration:none'>View filing</a>"
        "</td></tr>"
        for item in items
    )

    subject = "Insider filing alert - " + ", ".join(tickers_seen)
    footer_tickers = ", ".join(tickers_seen)

    html = (
        "<!DOCTYPE html><html lang='en'><body style='font-family:-apple-system,BlinkMacSystemFont,"
        "Segoe UI,sans-serif;color:#1f2937;max-width:600px;margin:0 auto;padding:32px 24px'>"
        "<h2 style='margin:0 0 4px'>New insider filings</h2>"
        "<p style='margin:0 0 20px;color:#6b7280'>Tickers you follow had new SEC Form 4 activity.</p>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
        "<thead><tr style='background:#f9fafb;text-align:left'>"
        "<th style='padding:8px 12px'>Ticker</th>"
        "<th style='padding:8px 12px'>Filed</th>"
        "<th style='padding:8px 12px'>Filing</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
        "<div style='margin-top:28px'>"
        "<a href='" + _APP_URL + "' style='display:inline-block;padding:10px 22px;background:#4f46e5;"
        "color:#fff;border-radius:6px;text-decoration:none;font-size:14px'>Manage your alerts</a>"
        "</div>"
        "<p style='margin-top:28px;color:#9ca3af;font-size:12px'>"
        "You are receiving this because you follow " + footer_tickers + " on StockAI."
        "</p></body></html>"
    )

    try:
        resend.Emails.send({
            "from": _ALERT_FROM,
            "to": email,
            "subject": subject,
            "html": html,
        })
        log.info("Alert sent to %s for %s", email, tickers_seen)
    except Exception as exc:
        log.error("Resend failed for %s: %s", email, exc)
