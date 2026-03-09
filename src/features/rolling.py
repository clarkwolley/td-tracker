"""
Player rolling feature engineering.

Computes rolling averages, TD streaks, scoring rates, and efficiency
metrics from weekly player stats.

Every feature is shifted by 1 game to prevent data leakage — we only
use information available BEFORE the game we're predicting.
"""

import pandas as pd
import numpy as np

from src.config import (
    ROLLING_WINDOW,
    MIN_GAMES_PLAYED,
    TD_ELIGIBLE_POSITIONS,
)

# Stat groups we compute rolling features for
_TD_COLS = ["passing_tds", "rushing_tds", "receiving_tds"]
_YARD_COLS = ["passing_yards", "rushing_yards", "receiving_yards"]
_VOLUME_COLS = ["attempts", "carries", "targets", "receptions", "completions"]
_EFFICIENCY_COLS = ["passing_epa", "rushing_epa", "receiving_epa"]

_ROLLING_COLS = _TD_COLS + _YARD_COLS + _VOLUME_COLS + _EFFICIENCY_COLS


def add_total_tds(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined TD count and binary scored_td flag."""
    df = df.copy()
    df["total_tds"] = df[_TD_COLS].fillna(0).sum(axis=1)
    df["scored_td"] = (df["total_tds"] > 0).astype(int)
    return df


def add_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative games-played count per player."""
    df = df.copy()
    df = df.sort_values(["player_id", "season", "week"])
    df["games_played"] = df.groupby("player_id").cumcount() + 1
    return df


def add_rolling_averages(
    df: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Add rolling averages for key stat columns.

    Each output column is named ``roll{window}_{original_col}``.
    Values are shifted by 1 so we never peek at the current game.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "season", "week"])

    all_cols = _ROLLING_COLS + (["total_tds"] if "total_tds" in df.columns else [])

    for col in all_cols:
        if col not in df.columns:
            continue
        # fillna(0): NaN means "didn't participate" (e.g. WR passing_epa)
        # which is semantically zero contribution, not missing data.
        roll_col = f"roll{window}_{col}"
        df[roll_col] = (
            df.groupby("player_id")[col]
            .transform(
                lambda s: s.fillna(0)
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        )

    return df


def add_td_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative TD scoring rate (% of prior games with at least 1 TD).

    Expanding window, shifted by 1 to prevent leakage.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "season", "week"])
    df["td_rate"] = (
        df.groupby("player_id")["scored_td"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return df


def add_td_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track consecutive-game TD streaks entering each game.

    - ``td_streak``:   consecutive games WITH a TD  (≥ 0)
    - ``cold_streak``: consecutive games WITHOUT one (≥ 0)
    """
    df = df.copy()
    df = df.sort_values(["player_id", "season", "week"])

    def _streak(series: pd.Series) -> pd.Series:
        shifted = series.shift(1)
        streaks = []
        current = 0
        for val in shifted:
            if pd.isna(val):
                streaks.append(0)
            elif val == 1:
                current = max(current, 0) + 1
                streaks.append(current)
            else:
                current = min(current, 0) - 1
                streaks.append(current)
        return pd.Series(streaks, index=series.index)

    raw = df.groupby("player_id")["scored_td"].transform(_streak)
    df["td_streak"] = raw.clip(lower=0)
    df["cold_streak"] = (-raw).clip(lower=0)
    return df


def add_opportunity_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiency features derived from rolling averages.

    - ``yards_per_carry``  — rushing yards / carries
    - ``yards_per_target`` — receiving yards / targets
    - ``td_per_touch``     — (rushing + receiving TDs) / (carries + targets)
    """
    df = df.copy()
    w = ROLLING_WINDOW

    # Yards per carry
    carries_col = f"roll{w}_carries"
    rush_yd_col = f"roll{w}_rushing_yards"
    if carries_col in df.columns and rush_yd_col in df.columns:
        df["yards_per_carry"] = np.where(
            df[carries_col] > 0, df[rush_yd_col] / df[carries_col], 0.0
        )

    # Yards per target
    tgt_col = f"roll{w}_targets"
    rec_yd_col = f"roll{w}_receiving_yards"
    if tgt_col in df.columns and rec_yd_col in df.columns:
        df["yards_per_target"] = np.where(
            df[tgt_col] > 0, df[rec_yd_col] / df[tgt_col], 0.0
        )

    # TD per touch (rushing + receiving)
    rush_td_col = f"roll{w}_rushing_tds"
    rec_td_col = f"roll{w}_receiving_tds"
    if all(c in df.columns for c in [rush_td_col, rec_td_col, carries_col, tgt_col]):
        touches = df[carries_col].fillna(0) + df[tgt_col].fillna(0)
        skill_tds = df[rush_td_col].fillna(0) + df[rec_td_col].fillna(0)
        df["td_per_touch"] = np.where(touches > 0, skill_tds / touches, 0.0)

    return df


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def compute_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full player-feature pipeline.

    Filters to TD-eligible positions, then layers on rolling averages,
    rates, streaks, and efficiency stats.  Drops players with fewer
    than ``MIN_GAMES_PLAYED`` games.
    """
    df = df[df["position"].isin(TD_ELIGIBLE_POSITIONS)].copy()

    df = add_total_tds(df)
    df = add_games_played(df)
    df = add_rolling_averages(df)
    df = add_td_rate(df)
    df = add_td_streak(df)
    df = add_opportunity_rates(df)

    df = df[df["games_played"] >= MIN_GAMES_PLAYED]
    return df
