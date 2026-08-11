"""Rank feature importance with SHAP.

Fits a random forest on the full sample and reports each feature's mean absolute
SHAP value (TreeExplainer). This is the descriptive importance analysis behind
the paper's importance figure; it is not part of the cross-validated prediction,
so treat the ordering as where the signal was concentrated, not exact ranks.
"""

import argparse
import numpy as np
import pandas as pd
import shap

from train import load, prep, apply_prep, build_model


def shap_importance(X, y, top=20):
    imp, sc = prep(X)
    Xs = apply_prep(imp, sc, X)

    rf = build_model("rf", len(np.unique(y)))
    rf.fit(Xs.values, y)

    values = shap.TreeExplainer(rf).shap_values(Xs.values)
    # shape depends on shap/sklearn version: a list per class, (n, f), or (n, f, classes)
    if isinstance(values, list):
        mean_abs = np.mean([np.abs(v).mean(axis=0) for v in values], axis=0)
    else:
        values = np.asarray(values)
        mean_abs = np.abs(values).mean(axis=(0, 2) if values.ndim == 3 else 0)

    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).head(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None, help="optional CSV for the ranked table")
    args = ap.parse_args()

    X, y = load(args.data, args.target)
    ranked = shap_importance(X, y, args.top)
    print(ranked.to_string())
    if args.out:
        ranked.rename("mean_abs_shap").to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
