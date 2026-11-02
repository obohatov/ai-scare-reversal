"""Train and compare models that score sharp falls.

Run from the repository root:

    python -m scripts.train

Settings are chosen on the validation period only. The chosen model is then
refitted on training and validation data and scores the test period. Those
scores are saved for the simulation, and no test metric is computed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from scripts.data_repo import DATA_DIR
from scripts.transform import DUMMIES, EVENTS_PATH, NUMERIC

PREDICTIONS_PATH = DATA_DIR / "processed" / "predictions.parquet"
REPORT_PATH = DATA_DIR.parent / "reports" / "models.json"
TARGET = "is_positive_rel_30d"
RANDOM_STATE = 42
TOP_SHARE = 0.10


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict[str, Any]


def candidates() -> list[Candidate]:
    """Small grids: few settings means less room to fit validation by luck."""
    grid: list[Candidate] = [
        Candidate("logistic", {"C": value}) for value in (0.01, 0.1, 1.0)
    ]
    grid += [
        Candidate("tree", {"max_depth": depth, "min_samples_leaf": 50})
        for depth in (3, 4, 5, 6, 8)
    ]
    grid += [
        Candidate(
            "forest",
            {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf},
        )
        for depth in (4, 6, 10)
        for leaf in (20, 50)
    ]
    grid += [
        Candidate(
            "xgboost",
            {
                "n_estimators": trees,
                "max_depth": depth,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.6,
                "min_child_weight": 20,
            },
        )
        for depth in (2, 3, 4)
        for trees in (100, 300)
    ]
    return grid


def make_model(candidate: Candidate) -> Any:
    """Build one unfitted model."""
    params = candidate.params
    if candidate.family == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, **params),
        )
    if candidate.family == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if candidate.family == "forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if candidate.family == "xgboost":
        return XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", **params
        )
    raise ValueError(f"Unknown model family: {candidate.family}")


def build_matrix(
    events: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Return numeric features plus dummies; `columns` fixes the training layout."""
    numeric = events[NUMERIC].astype("float32")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    dummies = pd.get_dummies(events[DUMMIES].astype(str), dtype="float32")
    matrix = pd.concat([numeric, dummies], axis=1)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0.0)
    return matrix


def top_share_metrics(
    scores: np.ndarray, frame: pd.DataFrame, share: float = TOP_SHARE
) -> dict[str, float]:
    """Measure the highest-scored events, which are the ones a strategy trades."""
    count = max(1, int(len(scores) * share))
    chosen = frame.iloc[np.argsort(scores)[::-1][:count]]
    return {
        "top_events": count,
        "top_precision": round(float(chosen[TARGET].mean()), 4),
        "top_median_rel_growth": round(float(chosen["rel_growth_30d"].median()), 4),
    }


def compare_models(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Fit every candidate on training data and score it on validation."""
    train_x = build_matrix(train)
    columns = list(train_x.columns)
    validation_x = build_matrix(validation, columns)
    rows = []
    for candidate in candidates():
        model = make_model(candidate)
        model.fit(train_x, train[TARGET].astype(int))
        scores = model.predict_proba(validation_x)[:, 1]
        rows.append(
            {
                "family": candidate.family,
                "params": json.dumps(candidate.params, sort_keys=True),
                "validation_auc": round(float(roc_auc_score(validation[TARGET], scores)), 4),
                **top_share_metrics(scores, validation),
            }
        )
        print(rows[-1], flush=True)
    return pd.DataFrame(rows).sort_values("validation_auc", ascending=False)


def fit_and_score(
    candidate: Candidate, fit_rows: pd.DataFrame, score_rows: pd.DataFrame
) -> np.ndarray:
    """Fit on some rows and return probabilities for others."""
    fit_x = build_matrix(fit_rows)
    model = make_model(candidate)
    model.fit(fit_x, fit_rows[TARGET].astype(int))
    return model.predict_proba(build_matrix(score_rows, list(fit_x.columns)))[:, 1]


def run(events: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Choose a model on validation, then score validation and test rows."""
    labelled = events[events[TARGET].notna()]
    train = labelled[labelled["split"] == "train"]
    validation = labelled[labelled["split"] == "validation"]
    table = compare_models(train, validation)
    best = table.iloc[0]
    chosen = Candidate(str(best["family"]), json.loads(best["params"]))

    validation_scored = validation.assign(
        score=fit_and_score(chosen, train, validation)
    )
    test = events[events["split"] == "test"]
    test_scored = test.assign(
        score=fit_and_score(chosen, pd.concat([train, validation]), test)
    )
    keep = [
        "date", "ticker", "split", "exit_date", "entry_open", "exit_close",
        "sector_growth_30d", "in_frozen_universe", "is_peer", "is_event_extra",
        "growth_30d", "rel_growth_30d", TARGET, "score",
    ]
    predictions = pd.concat([validation_scored[keep], test_scored[keep]], ignore_index=True)
    report = {
        "base_rate_validation": round(float(validation[TARGET].mean()), 4),
        "chosen": table.iloc[0].to_dict(),
        "best_per_family": table.groupby("family", sort=False).head(1).to_dict("records"),
        "all_candidates": table.to_dict("records"),
    }
    return report, predictions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    report, predictions = run(pd.read_parquet(EVENTS_PATH))
    predictions.to_parquet(PREDICTIONS_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("base_rate_validation", "best_per_family")}, indent=2))
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
