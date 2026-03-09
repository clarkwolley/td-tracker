"""
Prediction grading.

Compares saved predictions to actual results and generates
report cards — both for console output and Telegram.
"""

import numpy as np
import pandas as pd

from src.automation.storage import load_predictions, has_predictions


# ---------------------------------------------------------------------------
# Schedule analysis
# ---------------------------------------------------------------------------

def detect_last_played_week(schedules: pd.DataFrame) -> tuple[int, int] | None:
    """
    Find the most recent (season, week) where games have final scores.

    Returns None during deep offseason when no new games exist.
    """
    played = schedules.dropna(subset=["home_score", "away_score"])
    if played.empty:
        return None
    played = played.sort_values(["season", "week"])
    last = played.iloc[-1]
    return int(last["season"]), int(last["week"])


def detect_next_week(schedules: pd.DataFrame) -> tuple[int, int] | None:
    """
    Find the next (season, week) with scheduled but unplayed games.

    Returns None if every game on the schedule has been played
    (offseason / end of playoffs).
    """
    unplayed = schedules[schedules["home_score"].isna()]
    if unplayed.empty:
        return None
    unplayed = unplayed.sort_values(["season", "week"])
    first = unplayed.iloc[0]
    return int(first["season"]), int(first["week"])


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_week(
    season: int,
    week: int,
    feature_matrix: pd.DataFrame,
) -> dict | None:
    """
    Grade our saved predictions for (season, week) against actuals.

    Returns a dict with grading metrics, or None if we don't have
    saved predictions for that week or actuals aren't available yet.
    """
    saved = load_predictions(season, week)
    if saved is None:
        return None

    # Get actuals from the feature matrix
    actuals = feature_matrix[
        (feature_matrix["season"] == season) & (feature_matrix["week"] == week)
    ][["player_id", "scored_td"]].copy()

    if actuals.empty:
        return None

    # Drop stale scored_td from saved preds — actuals are ground truth
    saved = saved.drop(columns=["scored_td"], errors="ignore")

    merged = saved.merge(actuals, on="player_id", how="left")
    merged["scored_td"] = merged["scored_td"].fillna(0).astype(int)

    n = len(merged)
    hits = int(merged["scored_td"].sum())
    avg_prob = float(merged["td_prob"].mean())
    actual_rate = float(merged["scored_td"].mean())

    # Top-5 accuracy
    top5 = merged.head(5)
    top5_hits = int(top5["scored_td"].sum())

    # Best calls (high prob + scored) and biggest misses (high prob + didn't)
    scored = merged[merged["scored_td"] == 1].head(3)
    missed = merged[merged["scored_td"] == 0].head(3)

    return {
        "season": season,
        "week": week,
        "total": n,
        "hits": hits,
        "accuracy": hits / n if n else 0,
        "top5_hits": top5_hits,
        "top5_accuracy": top5_hits / min(5, n) if n else 0,
        "avg_predicted_prob": avg_prob,
        "actual_rate": actual_rate,
        "calibration_error": abs(avg_prob - actual_rate),
        "best_calls": scored[["player_name", "td_prob"]].to_dict("records"),
        "biggest_misses": missed[["player_name", "td_prob"]].to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_grade_report(grades: dict) -> str:
    """Format grades as a Telegram-friendly HTML message."""
    s, w = grades["season"], grades["week"]
    hits, total = grades["hits"], grades["total"]
    pct = grades["accuracy"]
    t5 = grades["top5_hits"]

    # Letter grade
    letter = _letter_grade(pct)

    lines = [
        f"<b>📊 Report Card — {s} Week {w}</b>",
        "",
        f"Grade: <b>{letter}</b>",
        f"Overall: <b>{hits}/{total}</b> hit ({pct:.0%})",
        f"Top 5:   <b>{t5}/5</b> ({grades['top5_accuracy']:.0%})",
        f"Calibration: predicted avg {grades['avg_predicted_prob']:.0%}, "
        f"actual {grades['actual_rate']:.0%} "
        f"(off by {grades['calibration_error']:.0%})",
    ]

    if grades["best_calls"]:
        lines.append("")
        lines.append("<b>🎯 Best calls:</b>")
        for p in grades["best_calls"]:
            lines.append(f"  ✅ {p['player_name']} ({p['td_prob']:.0%})")

    if grades["biggest_misses"]:
        lines.append("")
        lines.append("<b>😬 Biggest misses:</b>")
        for p in grades["biggest_misses"]:
            lines.append(f"  ❌ {p['player_name']} ({p['td_prob']:.0%})")

    return "\n".join(lines)


def print_grade_report(grades: dict) -> None:
    """Pretty-print grades to the console."""
    s, w = grades["season"], grades["week"]
    hits, total = grades["hits"], grades["total"]
    pct = grades["accuracy"]
    letter = _letter_grade(pct)

    print(f"\n📊 Report Card — {s} Week {w}")
    print(f"   Grade: {letter}")
    print(f"   Overall: {hits}/{total} ({pct:.0%})")
    print(f"   Top 5:   {grades['top5_hits']}/5 ({grades['top5_accuracy']:.0%})")
    print(f"   Calibration error: {grades['calibration_error']:.0%}")

    if grades["best_calls"]:
        print("   Best calls:")
        for p in grades["best_calls"]:
            print(f"     ✅ {p['player_name']} ({p['td_prob']:.0%})")

    if grades["biggest_misses"]:
        print("   Biggest misses:")
        for p in grades["biggest_misses"]:
            print(f"     ❌ {p['player_name']} ({p['td_prob']:.0%})")


def _letter_grade(accuracy: float) -> str:
    """Convert accuracy percentage to a letter grade."""
    if accuracy >= 0.80:
        return "A 🔥"
    if accuracy >= 0.65:
        return "B 👍"
    if accuracy >= 0.50:
        return "C 😐"
    if accuracy >= 0.35:
        return "D 😬"
    return "F 💀"
