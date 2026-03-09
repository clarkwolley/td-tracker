"""
nflverse data loader.

Pulls data directly from nflverse GitHub releases — the same source
that nfl_data_py wraps, but without the ancient pandas<2.0 dependency.

Data sources:
- Play-by-play:  detailed snap-level data for every NFL game
- Player stats:  weekly aggregated stats per player
- Schedules:     game schedule with scores, spreads, etc.
- Rosters:       weekly roster snapshots with positions, heights, etc.

All data is cached locally in data/cache/ to avoid re-downloading.
"""

import os
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import (
    NFLVERSE_PBP,
    NFLVERSE_PLAYER_STATS,
    NFLVERSE_SCHEDULES,
    NFLVERSE_ROSTERS,
    TRAINING_SEASONS,
)


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_TTL_HOURS = 12  # re-download after this many hours

log = logging.getLogger("td-tracker")


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(url: str) -> Path:
    """Generate a deterministic cache filename for a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    # Use a readable name + hash suffix
    name = url.split("/")[-1].replace(".parquet", "").replace(".csv", "")
    ext = ".parquet" if url.endswith(".parquet") else ".csv"
    return CACHE_DIR / f"{name}_{url_hash}{ext}"


def _is_cache_fresh(path: Path) -> bool:
    """Check if cached file exists and isn't stale."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _download(url: str, cache_path: Path) -> Path:
    """Download a file from nflverse and cache it locally."""
    import requests

    _ensure_cache_dir()
    log.info(f"   Downloading: {url.split('/')[-1]}")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    cache_path.write_bytes(resp.content)
    return cache_path


def _load_parquet(url: str) -> pd.DataFrame:
    """Load a parquet file from nflverse, using cache if fresh."""
    path = _cache_path(url)
    if not _is_cache_fresh(path):
        _download(url, path)
    return pd.read_parquet(path)


def _load_csv(url: str) -> pd.DataFrame:
    """Load a CSV file from nflverse, using cache if fresh."""
    path = _cache_path(url)
    if not _is_cache_fresh(path):
        _download(url, path)
    return pd.read_csv(path, low_memory=False)


# --- Public API ---


def get_player_stats() -> pd.DataFrame:
    """
    Load weekly player stats for all training seasons.

    Sources:
    - nflverse parquet for seasons it covers
    - ESPN API for any season nflverse is missing (e.g. current season)

    Returns a single DataFrame regardless of source mix.
    """
    print("📊 Loading player stats from nflverse...")
    df = _load_parquet(NFLVERSE_PLAYER_STATS)

    nflverse_seasons = set(df["season"].unique()) & set(TRAINING_SEASONS)
    missing_seasons = [s for s in TRAINING_SEASONS if s not in nflverse_seasons]

    df = df[df["season"].isin(TRAINING_SEASONS)]
    print(f"   {len(df):,} player-week rows, seasons {df['season'].min()}-{df['season'].max()}")

    # Backfill missing seasons from ESPN
    if missing_seasons:
        from src.data.espn import get_espn_player_stats

        print(f"   ⚠️  nflverse missing seasons: {missing_seasons} — falling back to ESPN")
        espn_frames = []
        for season in missing_seasons:
            espn_df = get_espn_player_stats(season)
            if not espn_df.empty:
                espn_frames.append(espn_df)
        if espn_frames:
            espn_all = pd.concat(espn_frames, ignore_index=True)
            df = pd.concat([df, espn_all], ignore_index=True)
            print(f"   📡 Combined: {len(df):,} total rows")

    return df


def get_schedules() -> pd.DataFrame:
    """
    Load NFL schedules with scores and game info.

    Columns include: game_id, season, week, gameday, home_team, away_team,
    home_score, away_score, spread_line, total_line, etc.
    """
    print("📅 Loading schedules from nflverse...")
    df = _load_csv(NFLVERSE_SCHEDULES)
    df = df[df["season"].isin(TRAINING_SEASONS)]
    print(f"   {len(df):,} games loaded")
    return df


def get_rosters(season: int) -> pd.DataFrame:
    """Load weekly roster data for a specific season."""
    url = NFLVERSE_ROSTERS.format(year=season)
    return _load_parquet(url)


def get_pbp(season: int) -> pd.DataFrame:
    """
    Load play-by-play data for a season.

    WARNING: These files are large (~100MB+ per season).
    Only use when you need snap-level detail.
    """
    url = NFLVERSE_PBP.format(year=season)
    print(f"📼 Loading {season} play-by-play (this is a big file)...")
    return _load_parquet(url)


def clear_cache() -> None:
    """Delete all cached nflverse data."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("🗑️  Cache cleared")
