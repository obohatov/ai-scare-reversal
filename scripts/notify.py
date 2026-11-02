"""Send short status messages to a Telegram chat."""

from __future__ import annotations

from typing import Any

import requests

from scripts.settings import Settings

SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4000


def send_message(session: Any, settings: Settings, text: str) -> bool:
    """Send a message; a missing token or a network failure never stops a run."""
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        return False
    url = SEND_URL.format(token=settings.telegram_bot_token)
    payload = {"chat_id": settings.telegram_chat_id, "text": text[:MAX_MESSAGE_LENGTH]}
    try:
        response = session.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True
