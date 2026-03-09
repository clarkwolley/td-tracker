"""
Feature-matrix builder.

Orchestrates the full pipeline: loads raw nflverse data, computes
player-level and game-context features, then produces a single
model-ready DataFrame.
"""

import pandas as pd

from src.config import ROLLING_WINDOW
from src.data.nflverse import get_player_stats, get_schedules
from src.features.rolling import compute_player_features
from src.features.context import (
    add_home_away,
    add_spread_and_total,
    compute_opponent_defense,
)


_w = ROLLING_WINDOW

# Columns the model trains on (position dummies added dynamically)
FEATURE_COLS = [
    # Rolling player stats
    f"roll{_w}_passing_tds",
    f"roll{_w}_rushing_tds",
    f"roll{_w}_receiving_tds",
    f"roll{_w}_total_tds",
    f"roll{_w}_passing_yards",
    f"roll{_w}_rushing_yards",
    f"roll{_w}_receiving_yards",
    f"roll{_w}_attempts",
    f"roll{_w}_carries",
    f"roll{_w}_targets",
    f"roll{_w}_receptions",
    f"roll{_w}_passing_epa",
    f"roll{_w}_rushing_epa",
    f"roll{_w}_receiving_epa",
    # Rates & efficiency
    "td_rate",
    "yards_per_carry",
    "yards_per_target",
    "td_per_touch",
    # Streaks
    "td_streak",
    "cold_streak",
    # Game context
    "is_home",
    "team_spread",
    "total_line",
    "implied_team_total",
    "opp_pts_allowed_avg",
]

TARGET_COL = "scored_td"

ID_COLS = ["player_id", "player_name", "season", "week", "recent_team"]


def build_feature_matrix() -> pd.DataFrame:
    """
    Build the complete feature matrix for TD prediction.

    Returns a DataFrame with ID columns, feature columns, and the
    binary target ``scored_td``.  Ready for train/test splitting.
    """
    # --- load raw data ---
    player_stats = get_player_stats()
    schedules = get_schedules()

    # --- player-level features (rolling, streaks, rates) ---
    print("⚙️  Computing player features...")
    df = compute_player_features(player_stats)

    # --- game-context features ---
    print("⚙️  Adding game context...")
    df = add_home_away(df, schedules)
    df = add_spread_and_total(df, schedules)

    # Opponent defensive quality (built from schedule scores only,
    # so we intentionally pass raw schedules — no filtered stats needed)
    opp_def = compute_opponent_defense(schedules)
    df = df.merge(
        opp_def,
        left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"],
        how="left",
    )
    df = df.drop(columns=["team"], errors="ignore")

    # --- one-hot encode position ---
    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=int)
    pos_cols = sorted(c for c in df.columns if c.startswith("pos_"))
    feature_cols = FEATURE_COLS + pos_cols

    # --- select & clean ---
    keep = ID_COLS + feature_cols + [TARGET_COL]
    available = [c for c in keep if c in df.columns]
    df = df[available].dropna(subset=[c for c in feature_cols if c in df.columns])

    print(f"✅ Feature matrix: {len(df):,} rows × {len(feature_cols)} features")
    print(f"   TD rate in dataset: {df[TARGET_COL].mean():.1%}")

    return df
