"""Automation and pipeline orchestration."""

from src.automation.pipeline import run_weekly, backtest_week
from src.automation.grading import grade_week, detect_last_played_week, detect_next_week
from src.automation.storage import save_predictions, load_predictions, list_saved

__all__ = [
    "run_weekly",
    "backtest_week",
    "grade_week",
    "detect_last_played_week",
    "detect_next_week",
    "save_predictions",
    "load_predictions",
    "list_saved",
]
