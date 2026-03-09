"""
Game-context feature engineering.

Adds situational information that affects TD probability:
home/away, Vegas lines, implied team total, and opponent
defensive quality.

Data sources are the nflverse schedule (scores, spreads, totals)
— no player-stats dependency, keeping concerns cleanly separated.
"""

import pandas as pd

from src.config import ROLLING_WINDOW


# ---------------------------------------------------------------------------
# Opponent defensive quality
# ---------------------------------------------------------------------------

def compute_opponent_defense(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling average of points allowed per team, mapped to opponents.

    Returns (season, week, team, opponent, opp_pts_allowed_avg).

    ``opp_pts_allowed_avg`` answers: *"How many points per game does my
    upcoming opponent typically give up?"*  High = soft defense = 🎉.
    """
    played = schedules.dropna(subset=["home_score", "away_score"]).copy()

    # Unstack schedule into one row per team per game
    home = played.assign(
        team=played["home_team"],
        opponent=played["away_team"],
        pts_allowed=played["away_score"],
    )[["season", "week", "team", "opponent", "pts_allowed"]]

    away = played.assign(
        team=played["away_team"],
        opponent=played["home_team"],
        pts_allowed=played["home_score"],
    )[["season", "week", "team", "opponent", "pts_allowed"]]

    team_games = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "season", "week"])
    )

    # Each team's own rolling defensive quality (shifted — no leakage)
    team_games["def_pts_allowed"] = (
        team_games.groupby("team")["pts_allowed"]
        .transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
    )

    # Lookup: (season, week, team) → that team's defensive quality
    def_lookup = team_games[["season", "week", "team", "def_pts_allowed"]].copy()

    # For each game row, grab the OPPONENT's defensive quality
    result = team_games[["season", "week", "team", "opponent"]].merge(
        def_lookup.rename(columns={
            "team": "opponent",
            "def_pts_allowed": "opp_pts_allowed_avg",
        }),
        on=["season", "week", "opponent"],
        how="left",
    )

    return result[["season", "week", "team", "opponent", "opp_pts_allowed_avg"]]


# ---------------------------------------------------------------------------
# Home / away
# ---------------------------------------------------------------------------

def add_home_away(
    player_stats: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Add binary ``is_home`` flag from the schedule."""
    home = schedules[["season", "week", "home_team"]].assign(is_home=1)
    home = home.rename(columns={"home_team": "recent_team"})

    away = schedules[["season", "week", "away_team"]].assign(is_home=0)
    away = away.rename(columns={"away_team": "recent_team"})

    lookup = pd.concat([home, away], ignore_index=True)
    return player_stats.merge(lookup, on=["season", "week", "recent_team"], how="left")


# ---------------------------------------------------------------------------
# Vegas lines
# ---------------------------------------------------------------------------

def add_spread_and_total(
    player_stats: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Add ``team_spread``, ``total_line``, and ``implied_team_total``.

    - ``team_spread``: from the player's team perspective (negative = favored).
    - ``total_line``:  Vegas over/under.
    - ``implied_team_total``: ``(total / 2) - (spread / 2)`` — how many
      points Vegas expects *this* team to score.
    """
    cols = ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    sched = schedules[cols].copy()

    home = sched.rename(columns={"home_team": "recent_team"}).copy()
    home["team_spread"] = home["spread_line"]

    away = sched.rename(columns={"away_team": "recent_team"}).copy()
    away["team_spread"] = -away["spread_line"]

    keep = ["season", "week", "recent_team", "team_spread", "total_line"]
    lookup = pd.concat([home[keep], away[keep]], ignore_index=True)

    df = player_stats.merge(lookup, on=["season", "week", "recent_team"], how="left")

    # Implied team total — the real gem for TD prediction
    df["implied_team_total"] = (df["total_line"] / 2) - (df["team_spread"] / 2)

    return df
