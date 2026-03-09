"""
ESPN API data loader.

Provides near-real-time player stats by scraping ESPN's public API.
Used when nflverse hasn't published stats for the current season yet.

Data flow:
    1. Get game IDs from nflverse schedule (has ESPN IDs)
    2. Fetch boxscore per game from ESPN summary API
    3. Parse into nflverse-compatible player stats DataFrame
    4. Map ESPN athlete IDs → nflverse gsis_ids via players table

No API key needed — ESPN's public site API is free and fast.
"""

import logging
from pathlib import Path
import time
from functools import lru_cache

import pandas as pd
import requests

from src.config import ESPN_API_BASE, TD_ELIGIBLE_POSITIONS

log = logging.getLogger("td-tracker")

_SUMMARY_URL = f"{ESPN_API_BASE}/summary"

# ESPN stat category → (label, our column name) mapping
_PASSING_MAP = {"YDS": "passing_yards", "TD": "passing_tds", "INT": "interceptions"}
_RUSHING_MAP = {"CAR": "carries", "YDS": "rushing_yards", "TD": "rushing_tds"}
_RECEIVING_MAP = {
    "REC": "receptions", "YDS": "receiving_yards",
    "TD": "receiving_tds", "TGTS": "targets",
}

# Position normalisation (ESPN uses various abbreviations)
_POS_NORMALIZE = {"WR": "WR", "RB": "RB", "QB": "QB", "TE": "TE", "FB": "RB"}


# ---------------------------------------------------------------------------
# ID crosswalk
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_players_table() -> pd.DataFrame:
    """Load the nflverse players table (cached for the process lifetime)."""
    url = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
    return pd.read_parquet(url)


def _load_id_crosswalk() -> dict[str, str]:
    """ESPN athlete ID → nflverse gsis_id.  Returns {espn_str: gsis_str}."""
    players = _load_players_table()
    return (
        players.dropna(subset=["espn_id", "gsis_id"])
        .assign(espn_id=lambda d: d["espn_id"].astype(int).astype(str))
        .set_index("espn_id")["gsis_id"]
        .to_dict()
    )


def _load_position_lookup() -> dict[str, str]:
    """ESPN athlete ID → position.  Returns {espn_str: position_str}."""
    players = _load_players_table()
    return (
        players.dropna(subset=["espn_id", "position"])
        .assign(espn_id=lambda d: d["espn_id"].astype(int).astype(str))
        .set_index("espn_id")["position"]
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Game data fetching
# ---------------------------------------------------------------------------

def _fetch_game_summary(game_id: int | str) -> dict:
    """Fetch ESPN game summary JSON for a single game."""
    resp = requests.get(_SUMMARY_URL, params={"event": str(game_id)}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_boxscore(
    summary: dict, season: int, week: int,
) -> list[dict]:
    """
    Parse an ESPN game summary into flat player stat rows.

    Returns a list of dicts, one per player, with nflverse-compatible
    column names.
    """
    boxscore = summary.get("boxscore", {})
    rows = []

    for team_group in boxscore.get("players", []):
        team_abbr = team_group["team"]["abbreviation"]

        # Build per-player stat dict from each category
        player_stats: dict[str, dict] = {}  # espn_id → stats

        for cat in team_group.get("statistics", []):
            cat_name = cat["name"]
            labels = cat.get("labels", [])

            if cat_name == "passing":
                col_map = _PASSING_MAP
            elif cat_name == "rushing":
                col_map = _RUSHING_MAP
            elif cat_name == "receiving":
                col_map = _RECEIVING_MAP
            else:
                continue

            for athlete_entry in cat.get("athletes", []):
                ath = athlete_entry["athlete"]
                espn_id = str(ath["id"])
                stats_list = athlete_entry.get("stats", [])

                if espn_id not in player_stats:
                    player_stats[espn_id] = {
                        "espn_id": espn_id,
                        "player_name": ath.get("displayName", ""),
                        "position": "",  # resolved after parsing
                        "_first_cat": cat_name,  # for fallback inference
                        "recent_team": team_abbr,
                        "season": season,
                        "week": week,
                    }

                for label, value in zip(labels, stats_list):
                    if cat_name == "passing" and label == "C/ATT":
                        parts = value.split("/")
                        player_stats[espn_id]["completions"] = _safe_int(parts[0])
                        player_stats[espn_id]["attempts"] = _safe_int(parts[1])
                    elif label in col_map:
                        player_stats[espn_id][col_map[label]] = _safe_int(value)

        rows.extend(player_stats.values())

    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ESPN_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def get_espn_player_stats(
    season: int, weeks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Fetch weekly player stats from ESPN for an entire season (or specific weeks).

    Results are cached as parquet per-season to avoid re-fetching 285 games.
    Delete ``data/cache/espn_stats_{season}.parquet`` to force a refresh.
    """
    # Check cache first (full-season only)
    if weeks is None:
        cache_path = _ESPN_CACHE_DIR / f"espn_stats_{season}.parquet"
        if cache_path.exists():
            print(f"📡 Loading {season} ESPN stats from cache...")
            return pd.read_parquet(cache_path)

    from src.data.nflverse import get_schedules

    print(f"📡 Fetching {season} player stats from ESPN (this takes a minute)...")
    schedules = get_schedules()
    sched = schedules[schedules["season"] == season].copy()

    if weeks:
        sched = sched[sched["week"].isin(weeks)]

    # Only games with ESPN IDs and final scores
    sched = sched.dropna(subset=["espn"])
    played = sched.dropna(subset=["home_score"])

    if played.empty:
        print("   ⚠️  No played games found")
        return pd.DataFrame()

    game_ids = played[["espn", "week"]].drop_duplicates()
    all_rows = []
    n_games = len(game_ids)

    for i, (_, row) in enumerate(game_ids.iterrows()):
        gid = int(row["espn"])
        wk = int(row["week"])
        try:
            summary = _fetch_game_summary(gid)
            game_rows = _parse_boxscore(summary, season, wk)
            all_rows.extend(game_rows)
        except Exception as e:
            log.warning("Failed to fetch game %s (wk %d): %s", gid, wk, e)

        # Be polite — don't hammer ESPN
        if (i + 1) % 16 == 0:
            print(f"   {i + 1}/{n_games} games fetched...")
            time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Map ESPN IDs → nflverse gsis_ids + positions
    crosswalk = _load_id_crosswalk()
    pos_lookup = _load_position_lookup()
    df["player_id"] = df["espn_id"].map(crosswalk)

    # Resolve positions: nflverse lookup → stat-category inference
    _CAT_TO_POS = {"passing": "QB", "rushing": "RB", "receiving": "WR"}
    df["position"] = df["espn_id"].map(pos_lookup).fillna(
        df["_first_cat"].map(_CAT_TO_POS)
    ).fillna("")
    df = df.drop(columns=["_first_cat"], errors="ignore")

    # Players without a crosswalk match get their ESPN ID prefixed
    unmapped = df["player_id"].isna()
    if unmapped.any():
        df.loc[unmapped, "player_id"] = "espn_" + df.loc[unmapped, "espn_id"]
        n_unmapped = unmapped.sum()
        log.info("%d players without nflverse ID mapping", n_unmapped)

    # Fill missing stat columns with 0 (nflverse convention)
    stat_cols = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
    ]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # EPA not available from ESPN — pipeline handles NaN gracefully
    for epa_col in ["passing_epa", "rushing_epa", "receiving_epa"]:
        df[epa_col] = float("nan")

    # Filter to skill positions
    df = df[df["position"].isin(TD_ELIGIBLE_POSITIONS)]

    df = df.drop(columns=["espn_id"])
    weeks_loaded = sorted(df["week"].unique())
    print(f"   ✅ {len(df):,} player-week rows, {n_games} games, weeks {weeks_loaded[0]}-{weeks_loaded[-1]}")

    # Cache full-season fetches
    if weeks is None:
        _ESPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _ESPN_CACHE_DIR / f"espn_stats_{season}.parquet"
        df.to_parquet(cache_path, index=False)
        print(f"   💾 Cached → {cache_path.name}")

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val: str) -> int:
    """Parse a stat value to int, handling floats and empty strings."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0
