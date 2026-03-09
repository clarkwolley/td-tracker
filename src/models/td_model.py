"""
TD prediction model.

Binary classifier: will a player score a TD in a given game?

Uses HistGradientBoosting — fast, handles mixed feature types well,
and doesn't need much hyperparameter babysitting out of the box.

Train/test split is temporal: older seasons train, most recent tests.
This mirrors real-world usage where we predict future games from
historical data.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)

from src.config import RANDOM_STATE
from src.features.builder import FEATURE_COLS, ID_COLS, TARGET_COL, build_feature_matrix

log = logging.getLogger("td-tracker")

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "td_model.joblib"


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame, test_season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by season — train on everything before ``test_season``.

    Defaults to the most recent season in the data (not the calendar
    year — the NFL season may not have started yet).

    This is the only honest split for time-series-ish sports data.
    Random splits leak future information and inflate metrics.
    """
    if test_season is None:
        test_season = int(df["season"].max())
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    return train, test


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Resolve feature columns that actually exist in the DataFrame."""
    pos_cols = sorted(c for c in df.columns if c.startswith("pos_"))
    base = [c for c in FEATURE_COLS if c != "position"]
    return [c for c in base + pos_cols if c in df.columns]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame | None = None) -> dict:
    """
    Train the TD prediction model and return evaluation metrics.

    Builds the feature matrix if not provided, does a temporal split,
    trains HistGradientBoosting, calibrates probabilities, evaluates,
    and saves the model to disk.

    Returns a dict of evaluation metrics on the test set.
    """
    if df is None:
        df = build_feature_matrix()

    train_df, test_df = temporal_split(df)
    feat_cols = _feature_cols(df)

    X_train, y_train = train_df[feat_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feat_cols], test_df[TARGET_COL]

    test_szn = int(test_df["season"].iloc[0])
    print(f"\n🏋️ Training on {len(train_df):,} rows (seasons < {test_szn})")
    print(f"🧪 Testing on  {len(test_df):,} rows (season {test_szn})")
    print(f"   Features: {len(feat_cols)}")

    # --- fit ---
    base_model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    base_model.fit(X_train, y_train)

    # --- calibrate probabilities (Platt scaling) ---
    # HistGBM probabilities can be slightly off; isotonic/sigmoid
    # calibration makes them trustworthy for "30% chance of TD" claims.
    model = CalibratedClassifierCV(
        base_model, method="isotonic", cv=5
    )
    model.fit(X_train, y_train)

    # --- evaluate ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = _evaluate(y_test, y_pred, y_prob)
    _print_report(metrics, y_test, y_pred)
    _print_top_features(model, X_test, y_test, feat_cols)

    # --- save ---
    _save_model(model, feat_cols)

    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute classification metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": (y_pred == y_true).mean(),
        "td_precision": (
            y_true[y_pred == 1].mean() if (y_pred == 1).any() else 0.0
        ),
        "td_recall": (
            y_pred[y_true == 1].mean() if (y_true == 1).any() else 0.0
        ),
    }


def _print_report(
    metrics: dict, y_true: pd.Series, y_pred: np.ndarray
) -> None:
    """Pretty-print evaluation results."""
    print("\n📊 Evaluation (test set):")
    print(f"   ROC-AUC:        {metrics['roc_auc']:.3f}")
    print(f"   Brier Score:    {metrics['brier_score']:.3f}  (lower is better)")
    print(f"   Log Loss:       {metrics['log_loss']:.3f}")
    print(f"   Accuracy:       {metrics['accuracy']:.1%}")
    print(f"   TD Precision:   {metrics['td_precision']:.1%}")
    print(f"   TD Recall:      {metrics['td_recall']:.1%}")
    report = classification_report(y_true, y_pred, target_names=["No TD", "TD"])
    print(f"\n{report}")


def _print_top_features(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    feat_cols: list[str], n: int = 10,
) -> None:
    """Show the most important features via permutation importance."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc",
    )
    indices = np.argsort(result.importances_mean)[::-1][:n]
    print(f"\n🔑 Top {n} features (permutation importance on test set):")
    for i, idx in enumerate(indices, 1):
        name = feat_cols[idx]
        mean = result.importances_mean[idx]
        std = result.importances_std[idx]
        print(f"   {i:2d}. {name:<30s} {mean:+.4f} ± {std:.4f}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_model(model, feat_cols: list[str]) -> None:
    """Save trained model + feature list to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "feature_cols": feat_cols}
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")


def load_model() -> tuple:
    """Load saved model and feature columns."""
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["feature_cols"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of player-game rows with TD probabilities.

    Expects the same feature columns used during training.
    Returns the input DataFrame with ``td_prob`` appended.
    """
    model, feat_cols = load_model()
    available = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(available)
    if missing:
        log.warning("Missing features (will be 0-filled): %s", missing)
        for col in missing:
            df[col] = 0

    df = df.copy()
    df["td_prob"] = model.predict_proba(df[feat_cols])[:, 1]
    return df
