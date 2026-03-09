"""
Telegram bot integration.

Sends TD predictions and alerts via the Telegram Bot API.

Setup:
1. Set ``TELEGRAM_BOT_TOKEN`` in ``.env``
2. Message the bot on Telegram (say "/start")
3. Run ``discover_chat_id()`` to detect your chat ID
4. Set ``TELEGRAM_CHAT_ID`` in ``.env``
5. Done — ``send_predictions()`` will message you directly.
"""

import logging
from pathlib import Path

import requests
import pandas as pd

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("td-tracker")

_API_BASE = "https://api.telegram.org/bot{token}"


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------

def _api_url(method: str) -> str:
    """Build a Telegram Bot API URL."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set — add it to your .env file"
        )
    return f"{_API_BASE.format(token=TELEGRAM_BOT_TOKEN)}/{method}"


def send_message(
    text: str,
    chat_id: str | None = None,
    parse_mode: str = "HTML",
) -> dict:
    """
    Send a text message to a Telegram chat.

    Uses ``TELEGRAM_CHAT_ID`` from config if ``chat_id`` is not provided.
    """
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID not set — run discover_chat_id() first, "
            "then add it to your .env"
        )

    resp = requests.post(
        _api_url("sendMessage"),
        json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()
    if not result.get("ok"):
        log.error("Telegram API error: %s", result)
    return result


# ---------------------------------------------------------------------------
# Chat ID discovery
# ---------------------------------------------------------------------------

def discover_chat_id() -> str | None:
    """
    Find your chat ID by reading recent messages sent to the bot.

    Steps:
    1. Open Telegram and message the bot (anything — "/start" works).
    2. Call this function.
    3. It returns the chat_id — add it to ``.env``.
    """
    resp = requests.get(_api_url("getUpdates"), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok") or not data.get("result"):
        print("❌ No messages found. Send /start to the bot first!")
        return None

    # Return the most recent chat ID
    for update in reversed(data["result"]):
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        chat_id = str(chat.get("id", ""))
        username = chat.get("username", "unknown")
        first = chat.get("first_name", "")
        if chat_id:
            print(f"✅ Found chat: {first} (@{username}) → chat_id: {chat_id}")
            print(f"   Add to .env:  TELEGRAM_CHAT_ID={chat_id}")
            _update_env_chat_id(chat_id)
            return chat_id

    print("❌ Couldn't extract a chat ID. Try messaging the bot again.")
    return None


def _update_env_chat_id(chat_id: str) -> None:
    """Write the discovered chat_id back into .env for convenience."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return

    content = env_path.read_text()
    if "TELEGRAM_CHAT_ID=" in content:
        lines = content.splitlines()
        lines = [
            f"TELEGRAM_CHAT_ID={chat_id}" if l.startswith("TELEGRAM_CHAT_ID=") else l
            for l in lines
        ]
        env_path.write_text("\n".join(lines) + "\n")
        print(f"   ✏️  Updated .env with TELEGRAM_CHAT_ID={chat_id}")


# ---------------------------------------------------------------------------
# Formatted prediction messages
# ---------------------------------------------------------------------------

def format_predictions(
    preds: pd.DataFrame,
    season: int,
    week: int,
    top_n: int = 20,
) -> str:
    """Format a predictions DataFrame as a Telegram-friendly HTML message."""
    top = preds.head(top_n)
    has_actual = "scored_td" in top.columns

    lines = [
        f"<b>🏈 TD Predictions — {season} Week {week}</b>",
        f"<i>Top {len(top)} most likely TD scorers</i>",
        "",
    ]

    for i, (_, row) in enumerate(top.iterrows(), 1):
        prob = row["td_prob"]
        name = row.get("player_name", "???")
        pos = row.get("position", "?")
        team = row.get("recent_team", "?")

        # Fire emoji for top 5, football for the rest
        icon = "🔥" if i <= 5 else "🏈"
        actual = ""
        if has_actual:
            actual = " ✅" if row["scored_td"] == 1 else " ❌"

        lines.append(f"{icon} <b>{i}.</b> {name} ({pos}, {team}) — <b>{prob:.0%}</b>{actual}")

    if has_actual:
        hits = top["scored_td"].sum()
        lines.append(f"\n<i>Accuracy: {hits}/{len(top)} hit ({hits/len(top):.0%})</i>")

    return "\n".join(lines)


def send_predictions(
    preds: pd.DataFrame,
    season: int,
    week: int,
    top_n: int = 20,
    chat_id: str | None = None,
) -> dict:
    """Format and send predictions to Telegram."""
    text = format_predictions(preds, season, week, top_n=top_n)
    return send_message(text, chat_id=chat_id)
