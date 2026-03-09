"""
Automation pipeline.

Full weekly workflow, designed to run every Tuesday morning after
Monday Night Football wraps up::

    python -m src.automation                  # full weekly run
    python -m src.automation --backtest 2024 10
    python -m src.automation --retrain --no-notify

Pipeline steps:
    1. Pull fresh nflverse data
    2. Grade last week's predictions (if any)
    3. Retrain model with latest data
    4. Predict the upcoming week
    5. Save predictions to disk (for next week's grading)
    6. Send grade report + new predictions to Telegram
"""

import logging
import time

from src.data.nflverse import get_schedules
from src.features.builder import build_feature_matrix
from src.models.td_model import train as train_model
from src.predictions.engine import get_top_predictions, print_predictions
from src.notifications.telegram import (
    send_message,
    send_predictions as tg_send_predictions,
    format_predictions,
)
from src.automation.storage import save_predictions, has_predictions
from src.automation.grading import (
    detect_last_played_week,
    detect_next_week,
    grade_week,
    format_grade_report,
    print_grade_report,
)

log = logging.getLogger("td-tracker")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_weekly(
    top_n: int = 20,
    force_retrain: bool = False,
    notify: bool = True,
) -> None:
    """
    Full weekly pipeline.

    1. Pull fresh data from nflverse
    2. Grade previous week's predictions
    3. Retrain model
    4. Predict upcoming week
    5. Save + notify
    """
    start = time.time()
    _banner("🐶 TD Tracker — Weekly Pipeline")

    # --- 1. Fresh data ---
    df = build_feature_matrix()
    schedules = get_schedules()

    last_played = detect_last_played_week(schedules)
    upcoming = detect_next_week(schedules)

    if last_played:
        lp_s, lp_w = last_played
        print(f"\n📅 Last played: {lp_s} Week {lp_w}")
    if upcoming:
        up_s, up_w = upcoming
        print(f"📅 Next week:   {up_s} Week {up_w}")
    else:
        print("📅 No upcoming games scheduled (offseason)")

    # --- 2. Grade previous predictions ---
    grade_report = _grade_step(last_played, df, notify)

    # --- 3. Retrain model ---
    print("\n🏋️ Retraining model with latest data...")
    train_model(df)

    # --- 4. Predict upcoming week ---
    if upcoming:
        _predict_step(upcoming, top_n, df, notify)
    else:
        msg = "🏈 Offseason — no upcoming games to predict. Model retrained and ready for kickoff!"
        print(f"\n{msg}")
        if notify:
            _safe_notify(lambda: send_message(msg))

    elapsed = time.time() - start
    print(f"\n⏱️  Pipeline complete in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def _grade_step(
    last_played: tuple[int, int] | None,
    df,
    notify: bool,
) -> dict | None:
    """Grade last week's predictions if we have them."""
    if last_played is None:
        print("\n📊 No completed games to grade")
        return None

    season, week = last_played

    if not has_predictions(season, week):
        print(f"\n📊 No saved predictions for {season} W{week} — nothing to grade")
        return None

    print(f"\n📊 Grading {season} Week {week} predictions...")
    grades = grade_week(season, week, df)

    if grades is None:
        print("   ⚠️  Couldn't match predictions to actuals")
        return None

    print_grade_report(grades)

    if notify:
        report_text = format_grade_report(grades)
        _safe_notify(lambda: send_message(report_text))

    return grades


def _predict_step(
    upcoming: tuple[int, int],
    top_n: int,
    df,
    notify: bool,
) -> None:
    """Generate, save, and send predictions for the upcoming week."""
    season, week = upcoming
    print(f"\n🎯 Predicting {season} Week {week}...")

    preds = get_top_predictions(season, week, n=top_n, df=df)

    if preds.empty:
        print("   ⚠️  No predictions generated (schedule/data issue)")
        return

    print_predictions(preds, title=f"🏈 {season} Week {week} — Top {top_n}")

    # Save for next week's grading
    save_predictions(preds, season, week)

    if notify:
        _safe_notify(
            lambda: tg_send_predictions(preds, season, week, top_n=top_n)
        )
        print("📱 Predictions sent to Telegram!")


# ---------------------------------------------------------------------------
# Backtest utility
# ---------------------------------------------------------------------------

def backtest_week(
    season: int,
    week: int,
    top_n: int = 20,
    notify: bool = False,
) -> None:
    """
    Run predictions on a historical week and compare to actuals.

    Optionally saves predictions and sends grade + predictions
    to Telegram (useful for demo/testing the full flow).
    """
    _banner(f"🔬 Backtest — {season} Week {week}")
    df = build_feature_matrix()
    preds = get_top_predictions(season, week, n=top_n, df=df)

    if preds.empty:
        print("   No data for that week.")
        return

    print_predictions(preds, title=f"🔬 Backtest — {season} Week {week}")

    if "scored_td" in preds.columns:
        hits = int(preds["scored_td"].sum())
        total = len(preds)
        print(f"\n📊 {hits}/{total} predicted players actually scored ({hits/total:.0%})")

    if notify:
        _safe_notify(
            lambda: tg_send_predictions(preds, season, week, top_n=top_n)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def _safe_notify(fn) -> None:
    """Call a notification function, swallowing errors gracefully."""
    try:
        fn()
    except Exception as e:
        print(f"   ⚠️  Telegram failed: {e}")
        log.exception("Telegram notification error")
