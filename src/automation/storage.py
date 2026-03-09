"""
Prediction storage.

Persists weekly predictions as CSV files so we can grade them once
actual results arrive.  No database, no fuss — just flat files.

Layout::

    data/predictions/
        2024_W01.csv
        2024_W02.csv
        ...
"""

from pathlib import Path

import pandas as pd

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "predictions"


def save_predictions(preds: pd.DataFrame, season: int, week: int) -> Path:
    """Save a predictions DataFrame to disk. Returns the file path."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(season, week)
    preds.to_csv(path, index=False)
    print(f"💾 Predictions saved → {path.name}")
    return path


def load_predictions(season: int, week: int) -> pd.DataFrame | None:
    """Load saved predictions for a (season, week), or None if absent."""
    path = _path_for(season, week)
    if not path.exists():
        return None
    return pd.read_csv(path)


def has_predictions(season: int, week: int) -> bool:
    """Check if we have saved predictions for a given week."""
    return _path_for(season, week).exists()


def list_saved() -> list[tuple[int, int]]:
    """Return all (season, week) pairs with saved predictions."""
    if not PREDICTIONS_DIR.exists():
        return []
    results = []
    for f in sorted(PREDICTIONS_DIR.glob("*_W*.csv")):
        try:
            stem = f.stem  # e.g. "2024_W17"
            season, week_str = stem.split("_W")
            results.append((int(season), int(week_str)))
        except ValueError:
            continue
    return results


def _path_for(season: int, week: int) -> Path:
    return PREDICTIONS_DIR / f"{season}_W{week:02d}.csv"
