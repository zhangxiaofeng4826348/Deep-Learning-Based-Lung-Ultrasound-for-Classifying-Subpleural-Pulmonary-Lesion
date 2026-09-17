#!/usr/bin/env python
"""Logistic-regression baseline utilities for clinical or ultrasound-feature models.

Input CSV format:
    label,<feature_1>,<feature_2>,...

Continuous/categorical preprocessing should be performed before running this script, or
implemented externally according to the analysis plan. This utility is intended to document the
baseline comparison workflow used in the manuscript.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slla_unet.metrics import classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Logistic-regression baseline")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out-dir", default="outputs/baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    if args.label_col not in train.columns or args.label_col not in test.columns:
        raise ValueError(f"label column not found: {args.label_col}")
    feature_cols = [c for c in train.columns if c != args.label_col]
    if not feature_cols:
        raise ValueError("No feature columns found")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    model.fit(train[feature_cols], train[args.label_col].astype(int))
    prob = model.predict_proba(test[feature_cols])[:, 1]
    metrics = asdict(classification_metrics(test[args.label_col].astype(int).values, prob, threshold=0.5))
    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
