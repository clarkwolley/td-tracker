"""
Tests for the features module.

Uses small synthetic DataFrames so we never hit the network.
"""

import pandas as pd
import numpy as np
import pytest

from src.features.rolling import (
    add_total_tds,
    add_games_played,
    add_rolling_averages,
    add_td_rate,
    add_td_streak,
    add_opportunity_rates,
    compute_player_features,
)
from src.features.context import (
    compute_opponent_defense,
    add_home_away,
    add_spread_and_total,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player_stats(n_weeks: int = 8) -> pd.DataFrame:
    """Minimal player-stats DataFrame for one player."""
    return pd.DataFrame({
        "player_id": ["P1"] * n_weeks,
        "player_name": ["Test Player"] * n_weeks,
        "recent_team": ["KC"] * n_weeks,
        "position": ["WR"] * n_weeks,
        "season": [2024] * n_weeks,
        "week": list(range(1, n_weeks + 1)),
        "passing_tds": [0] * n_weeks,
        "rushing_tds": [0, 0, 1, 0, 1, 1, 0, 0],
        "receiving_tds": [1, 0, 0, 1, 0, 1, 0, 1],
        "passing_yards": [0] * n_weeks,
        "rushing_yards": [12, 8, 25, 5, 30, 22, 10, 15],
        "receiving_yards": [85, 42, 60, 110, 55, 90, 30, 75],
        "attempts": [0] * n_weeks,
        "carries": [3, 2, 5, 1, 6, 4, 2, 3],
        "targets": [8, 5, 6, 10, 7, 9, 4, 8],
        "receptions": [5, 3, 4, 7, 4, 6, 2, 5],
        "completions": [0] * n_weeks,
        "passing_epa": [0.0] * n_weeks,
        "rushing_epa": [0.1, -0.2, 0.5, -0.1, 0.6, 0.3, -0.3, 0.1],
        "receiving_epa": [1.2, 0.3, 0.5, 1.8, 0.4, 1.1, -0.2, 0.9],
    })


def _make_schedules() -> pd.DataFrame:
    """Minimal schedule with 8 weeks, 2 teams."""
    rows = []
    for wk in range(1, 9):
        rows.append({
            "season": 2024,
            "week": wk,
            "home_team": "KC" if wk % 2 else "BUF",
            "away_team": "BUF" if wk % 2 else "KC",
            "home_score": 24 + wk,
            "away_score": 20 - wk,
            "spread_line": -3.5 if wk % 2 else 3.5,
            "total_line": 47.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling feature tests
# ---------------------------------------------------------------------------

class TestTotalTds:
    def test_adds_columns(self):
        df = add_total_tds(_make_player_stats())
        assert "total_tds" in df.columns
        assert "scored_td" in df.columns

    def test_total_is_sum(self):
        df = add_total_tds(_make_player_stats())
        expected = df["passing_tds"] + df["rushing_tds"] + df["receiving_tds"]
        pd.testing.assert_series_equal(df["total_tds"], expected, check_names=False)

    def test_scored_td_is_binary(self):
        df = add_total_tds(_make_player_stats())
        assert set(df["scored_td"].unique()).issubset({0, 1})


class TestGamesPlayed:
    def test_monotonic(self):
        df = add_games_played(_make_player_stats())
        assert df["games_played"].is_monotonic_increasing

    def test_starts_at_one(self):
        df = add_games_played(_make_player_stats())
        assert df["games_played"].iloc[0] == 1


class TestRollingAverages:
    def test_no_leakage(self):
        """Week 1 should have NaN rolling values (nothing to look back at)."""
        df = add_total_tds(_make_player_stats())
        df = add_rolling_averages(df, window=4)
        week1 = df[df["week"] == 1].iloc[0]
        assert pd.isna(week1["roll4_rushing_tds"])

    def test_shifted_correctly(self):
        """Week 2's rolling value should equal week 1's actual value."""
        df = add_total_tds(_make_player_stats())
        df = add_rolling_averages(df, window=4)
        df = df.sort_values("week")
        # With window=4, min_periods=1, week 2's roll = just week 1's value
        assert df.iloc[1]["roll4_receiving_tds"] == df.iloc[0]["receiving_tds"]


class TestTdRate:
    def test_no_leakage(self):
        df = add_total_tds(_make_player_stats())
        df = add_td_rate(df)
        assert pd.isna(df[df["week"] == 1].iloc[0]["td_rate"])

    def test_rate_in_range(self):
        df = add_total_tds(_make_player_stats())
        df = add_td_rate(df)
        valid = df["td_rate"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


class TestTdStreak:
    def test_non_negative(self):
        df = add_total_tds(_make_player_stats())
        df = add_td_streak(df)
        assert (df["td_streak"] >= 0).all()
        assert (df["cold_streak"] >= 0).all()

    def test_mutually_exclusive(self):
        """Can't have both a hot and cold streak at the same time."""
        df = add_total_tds(_make_player_stats())
        df = add_td_streak(df)
        both_positive = (df["td_streak"] > 0) & (df["cold_streak"] > 0)
        assert not both_positive.any()


class TestOpportunityRates:
    def test_no_division_by_zero(self):
        df = add_total_tds(_make_player_stats())
        df = add_rolling_averages(df)
        df = add_opportunity_rates(df)
        assert not df["yards_per_carry"].isna().any()
        assert not df["yards_per_target"].isna().any()
        assert np.isfinite(df["td_per_touch"]).all()


# ---------------------------------------------------------------------------
# Context feature tests
# ---------------------------------------------------------------------------

class TestHomeAway:
    def test_binary(self):
        stats = _make_player_stats()
        schedules = _make_schedules()
        df = add_home_away(stats, schedules)
        assert set(df["is_home"].dropna().unique()).issubset({0, 1})


class TestSpreadAndTotal:
    def test_implied_total_positive(self):
        stats = _make_player_stats()
        schedules = _make_schedules()
        df = add_spread_and_total(stats, schedules)
        valid = df["implied_team_total"].dropna()
        assert (valid > 0).all()

    def test_spread_flipped_for_away(self):
        """Away team spread should be opposite sign of home spread."""
        schedules = _make_schedules()
        # Week 1: KC is home, spread = -3.5 (KC favored)
        stats_home = pd.DataFrame({
            "player_id": ["H1"], "season": [2024], "week": [1],
            "recent_team": ["KC"],
        })
        stats_away = pd.DataFrame({
            "player_id": ["A1"], "season": [2024], "week": [1],
            "recent_team": ["BUF"],
        })
        h = add_spread_and_total(stats_home, schedules)
        a = add_spread_and_total(stats_away, schedules)
        assert h.iloc[0]["team_spread"] == -a.iloc[0]["team_spread"]


class TestOpponentDefense:
    def test_returns_expected_columns(self):
        schedules = _make_schedules()
        result = compute_opponent_defense(schedules)
        assert "opp_pts_allowed_avg" in result.columns
        assert "opponent" in result.columns

    def test_no_leakage_week1(self):
        """Week 1 opponent defense should be NaN (no prior games)."""
        schedules = _make_schedules()
        result = compute_opponent_defense(schedules)
        week1 = result[result["week"] == 1]
        assert week1["opp_pts_allowed_avg"].isna().all()


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestComputePlayerFeatures:
    def test_filters_positions(self):
        stats = _make_player_stats()
        # Add a kicker — should get dropped
        kicker = stats.iloc[:2].copy()
        kicker["player_id"] = "K1"
        kicker["position"] = "K"
        combined = pd.concat([stats, kicker], ignore_index=True)

        result = compute_player_features(combined)
        assert "K" not in result["position"].values

    def test_min_games_filter(self):
        result = compute_player_features(_make_player_stats())
        assert (result["games_played"] >= 3).all()

    def test_has_all_feature_columns(self):
        result = compute_player_features(_make_player_stats())
        expected = [
            "total_tds", "scored_td", "games_played",
            "td_rate", "td_streak", "cold_streak",
            "yards_per_carry", "yards_per_target", "td_per_touch",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"
