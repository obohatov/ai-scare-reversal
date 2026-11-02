"""Project settings. Secrets come from the environment or a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
APP_NAME = "ai-scare-reversal"
APP_VERSION = "0.1"


@dataclass(frozen=True)
class Settings:
    """Values read from the environment. Empty strings mean "not configured"."""

    contact_name: str = ""
    contact_email: str = ""
    telegram_bot_token: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""

    @property
    def has_contact(self) -> bool:
        return bool(self.contact_name and self.contact_email)

    @property
    def secrets(self) -> tuple[str, ...]:
        """Values that must never reach a report, a log line or Git."""
        values = (
            self.contact_email,
            self.telegram_bot_token,
            self.alpaca_api_key,
            self.alpaca_secret_key,
        )
        return tuple(value for value in values if value)


def load_settings() -> Settings:
    """Read settings from a local .env file and the process environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        contact_name=os.getenv("CONTACT_NAME", "").strip(),
        contact_email=os.getenv("CONTACT_EMAIL", "").strip(),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", "").strip(),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip(),
    )
