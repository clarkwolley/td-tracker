"""
TD Tracker configuration.

Centralised settings for data sources, seasons, model parameters,
and notification credentials.

Secrets (API tokens, chat IDs) are loaded from a ``.env`` file at
the project root — never committed to version control.
"""

import os
from datetime import datetime as _dt
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# NFL season detection
# ---------------------------------------------------------------------------
# NFL seasons span two calendar years (Sep-Feb).
# Jan-Aug → current season is last year.  Sep-Dec → this year.
CURRENT_SEASON = _dt.now().year if _dt.now().month >= 9 else _dt.now().year - 1

# ---------------------------------------------------------------------------
# nflverse data — the gold standard for NFL analytics
# ---------------------------------------------------------------------------
NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLVERSE_PBP = f"{NFLVERSE_BASE}/pbp/play_by_play_{{year}}.parquet"
NFLVERSE_PLAYER_STATS = f"{NFLVERSE_BASE}/player_stats/player_stats.parquet"
NFLVERSE_SCHEDULES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFLVERSE_ROSTERS = f"{NFLVERSE_BASE}/weekly_rosters/roster_weekly_{{year}}.parquet"

# ESPN API — for live scores and today's schedule
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# Seasons to collect for training
TRAINING_SEASONS = list(range(2021, CURRENT_SEASON + 1))

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 4         # games for rolling averages (~quarter season)
MIN_GAMES_PLAYED = 3       # minimum games to include a player
STREAK_MIN_GAMES = 2       # consecutive games to count as a "streak"

# TD rate thresholds
HIGH_TD_THRESHOLD = 0.15   # TD scoring rate above this = high-usage scorer

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Recency weighting — adjusted for NFL's shorter season
RECENCY_FULL_WEIGHT_GAMES = 17    # ~1 full NFL season
RECENCY_MIN_WEIGHT = 0.20         # oldest data floor
RECENCY_DECAY_RATE = 0.03         # faster decay (fewer games = each matters more)

# Retrain triggers
RETRAIN_INTERVAL_DAYS = 7     # retrain weekly during season

# ---------------------------------------------------------------------------
# Position groups for TD prediction
# ---------------------------------------------------------------------------
TD_ELIGIBLE_POSITIONS = ["QB", "RB", "WR", "TE"]

# Home field advantage in NFL (~57% home win rate historically)
HOME_FIELD_ADVANTAGE = 0.57

# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
