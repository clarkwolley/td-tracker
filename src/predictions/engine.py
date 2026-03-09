"""
Prediction engine.

Scores players for a given (season, week) with TD probabilities.

Works in two modes:
- **Historical**: week already played → features come from the pipeline.
- **Upcoming**: week hasn't happened yet → carries forward each player's
  latest rolling features and merges fresh game-context from the schedule.
"""

import logging

import pandas as pd

from src.config import TD_ELIGIBLE_POSITIONS
from src.data.nflverse import get_player_stats, get_schedules
from src.features.builder import build_feature_matrix
from src.features.context import (
    add_home_away,
    add_spread_and_total,
    compute_opponent_defense,
)
from src.features.rolling import compute_player_features
from src.models.td_model import load_model, _feature_cols

log = logging.getLogger("td-tracker")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_week(
    season: int,
    week: int,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score every eligible player for a given (season, week).

    Returns a DataFrame sorted by ``td_prob`` descending, with columns:
    player_id, player_name, recent_team, position, td_prob, and
    (for historical weeks) scored_td so you can check accuracy.
    """
    if df is None:
        df = build_feature_matrix()

    model, feat_cols = load_model()
    week_df = df[(df["season"] == season) & (df["week"] == week)]

    if week_df.empty:
        print(f"📡 Week {week} not in historical data — building from latest stats...")
        week_df = _build_upcoming_week(df, season, week)

    if week_df.empty:
        print(f"⚠️  No data for season {season} week {week}")
        return pd.DataFrame()

    week_df = week_df.copy()
    available = [c for c in feat_cols if c in week_df.columns]
    week_df["td_prob"] = model.predict_proba(week_df[available])[:, 1]

    # Attach position back from dummies for display
    pos_cols = [c for c in week_df.columns if c.startswith("pos_")]
    if pos_cols and "position" not in week_df.columns:
        week_df["position"] = (
            week_df[pos_cols]
            .idxmax(axis=1)
            .str.replace("pos_", "", regex=False)
        )

    display_cols = ["player_name", "position", "recent_team", "td_prob"]
    if "scored_td" in week_df.columns:
        display_cols.append("scored_td")
    extra = [c for c in display_cols if c in week_df.columns]

    return week_df[["player_id"] + extra].sort_values("td_prob", ascending=False)


def get_top_predictions(
    season: int,
    week: int,
    n: int = 20,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-N players most likely to score a TD."""
    preds = predict_week(season, week, df=df)
    return preds.head(n)


def print_predictions(preds: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a predictions DataFrame to the console."""
    if preds.empty:
        print("No predictions to show.")
        return

    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")

    has_actual = "scored_td" in preds.columns
    header = f"{'#':>3}  {'Player':<22} {'Pos':<4} {'Team':<5} {'Prob':>6}"
    if has_actual:
        header += f"  {'Actual':>6}"
    print(header)
    print("-" * len(header))

    for i, (_, row) in enumerate(preds.iterrows(), 1):
        line = (
            f"{i:>3}. {row['player_name']:<22} "
            f"{row.get('position', '?'):<4} "
            f"{row['recent_team']:<5} "
            f"{row['td_prob']:>5.1%}"
        )
        if has_actual:
            actual = "✅" if row["scored_td"] == 1 else "❌"
            line += f"  {actual:>6}"
        print(line)


# ---------------------------------------------------------------------------
# Upcoming-week construction
# ---------------------------------------------------------------------------

def _build_upcoming_week(
    historical_df: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """
    Build feature rows for a week that hasn't been played yet.

    Takes each active player's latest rolling features and merges
    with the upcoming schedule for game context.
    """
    schedules = get_schedules()
    sched_week = schedules[
        (schedules["season"] == season) & (schedules["week"] == week)
    ]
    if sched_week.empty:
        log.warning("No schedule found for season %d week %d", season, week)
        return pd.DataFrame()

    # Teams playing this week
    playing_teams = set(sched_week["home_team"]) | set(sched_week["away_team"])

    # Each player's most recent row (carries their rolling features forward)
    latest = (
        historical_df
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)
        .copy()
    )
    latest = latest[latest["recent_team"].isin(playing_teams)]

    # Overwrite season/week to the target
    latest["season"] = season
    latest["week"] = week

    # Re-merge game context for the new week
    latest = latest.drop(
        columns=["is_home", "team_spread", "total_line",
                 "implied_team_total", "opp_pts_allowed_avg", "opponent"],
        errors="ignore",
    )
    latest = add_home_away(latest, schedules)
    latest = add_spread_and_total(latest, schedules)

    opp_def = compute_opponent_defense(schedules)
    latest = latest.merge(
        opp_def,
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left",
    )
    latest = latest.drop(columns=["team"], errors="ignore")

    # Drop target — we don't know it yet
    latest = latest.drop(columns=["scored_td"], errors="ignore")

    return latest
