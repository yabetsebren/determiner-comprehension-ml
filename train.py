"""
Train and evaluate the determiner-comprehension classifiers.

Reads a de-identified feature table (one row per child) and a target column,
then runs the nested cross-validation described in the paper: everything that
touches the labels -- imputation, scaling, and mRMR feature selection -- is fit
inside each training fold only. Significance comes from a label-permutation test
on the same pipeline, not from a parametric assumption.

Usage:
    python train.py --data features.csv --target group --model rf
    python train.py --data features.csv --target lang_impaired --model enet --perms 1000

The feature table is expected to have an "id" column, the target column, and
the numeric feature columns. Anything else is ignored.
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

SEED = 42
K_GRID = (4, 6, 8)          # candidate feature-set sizes, tuned by inner CV
N_REPEATS = 10
N_FOLDS = 5


def build_model(name, n_classes):
    """Return an unfitted classifier. Class weights handle the uneven groups."""
    if name == "enet":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                  C=0.1, class_weight="balanced", max_iter=5000,
                                  random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2,
                                      class_weight="balanced", random_state=SEED)
    if name == "xgb":
        if XGBClassifier is None:
            raise SystemExit("xgboost is not installed (pip install xgboost)")
        obj = "multi:softmax" if n_classes > 2 else "binary:logistic"
        return XGBClassifier(objective=obj, n_estimators=200, max_depth=3,
                             learning_rate=0.1, verbosity=0, random_state=SEED)
    raise ValueError(f"unknown model: {name}")


def mrmr(X, y, k):
    """mRMR (Peng et al., 2005), MID form: greedily keep features that are
    relevant to y but not redundant with the ones already picked."""
    cols = list(X.columns)
    if len(cols) <= k:
        return cols
    relevance = pd.Series(mutual_info_classif(X.values, y, random_state=SEED), index=cols)
    chosen = [relevance.idxmax()]
    redundancy = pd.Series(0.0, index=cols)
    while len(chosen) < k:
        rest = [c for c in cols if c not in chosen]
        redundancy[rest] += mutual_info_regression(
            X[rest].values, X[chosen[-1]].values, random_state=SEED)
        score = relevance[rest] - redundancy[rest] / len(chosen)
        chosen.append(score.idxmax())
    return chosen


def prep(train_X):
    """Fit imputer + scaler on the training fold only."""
    imp = SimpleImputer(strategy="median").fit(train_X.values)
    sc = StandardScaler().fit(imp.transform(train_X.values))
    return imp, sc


def apply_prep(imp, sc, X):
    return pd.DataFrame(sc.transform(imp.transform(X.values)), columns=X.columns)


def fit_predict(name, Xtr, ytr, Xte):
    model = build_model(name, len(np.unique(ytr)))
    if name == "xgb":
        model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        model.fit(Xtr, ytr)
    return model.predict(Xte)


def choose_k(X, y, name):
    """Pick the feature-set size by a short inner CV (macro-F1). mRMR is
    prefix-nested, so we select once at max(k) per inner fold and score prefixes."""
    if len(K_GRID) == 1:
        return K_GRID[0]
    inner = StratifiedKFold(n_splits=min(3, np.bincount(y).min()),
                            shuffle=True, random_state=SEED)
    tally = {k: [] for k in K_GRID}
    for itr, ite in inner.split(X.values, y):
        imp, sc = prep(X.iloc[itr])
        Xi = apply_prep(imp, sc, X.iloc[itr])
        Xj = apply_prep(imp, sc, X.iloc[ite])
        order = mrmr(Xi, y[itr], max(K_GRID))
        for k in K_GRID:
            pred = fit_predict(name, Xi[order[:k]].values, y[itr], Xj[order[:k]].values)
            tally[k].append(f1_score(y[ite], pred, average="macro"))
    return max(K_GRID, key=lambda k: np.mean(tally[k]))


def cross_validate(X, y, name, repeats=N_REPEATS):
    """Nested repeated stratified CV. Returns fold-level macro-F1 and balanced acc."""
    n_splits = min(N_FOLDS, np.bincount(y).min())
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    macro, bal = [], []
    for tr, te in cv.split(X.values, y):
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
        imp, sc = prep(Xtr)
        Xtr_s, Xte_s = apply_prep(imp, sc, Xtr), apply_prep(imp, sc, Xte)
        feats = mrmr(Xtr_s, ytr, choose_k(Xtr_s, ytr, name))
        pred = fit_predict(name, Xtr_s[feats].values, ytr, Xte_s[feats].values)
        macro.append(f1_score(y[te], pred, average="macro"))
        bal.append(balanced_accuracy_score(y[te], pred))
    return np.array(macro), np.array(bal)


def permutation_test(X, y, name, observed, n_perms=1000):
    """Shuffle the labels, rerun the whole pipeline, count how often chance beats us."""
    rng = np.random.default_rng(SEED)
    ge = 0
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        null_macro, _ = cross_validate(X, y_perm, name, repeats=2)  # fewer repeats under the null
        if null_macro.mean() >= observed:
            ge += 1
        if (i + 1) % 100 == 0:
            print(f"  permutation {i+1}/{n_perms}", flush=True)
    return (ge + 1) / (n_perms + 1)


def load(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"target '{target}' not in {path}; columns: {list(df.columns)}")
    df = df[df[target].notna()].reset_index(drop=True)
    drop = {c for c in ("id", "subject", "sid") if c in df.columns} | {target}
    X = df.drop(columns=list(drop)).select_dtypes("number")
    y = pd.factorize(df[target])[0]          # encode labels 0..K-1
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="feature table (CSV)")
    ap.add_argument("--target", required=True, help="label column to predict")
    ap.add_argument("--model", default="rf", choices=["enet", "rf", "xgb"])
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--out", default=None, help="optional CSV to append the result to")
    args = ap.parse_args()

    X, y = load(args.data, args.target)
    print(f"{X.shape[0]} children, {X.shape[1]} features, {len(np.unique(y))} classes")

    macro, bal = cross_validate(X, y, args.model)
    lo, hi = np.percentile(macro, [2.5, 97.5])
    print(f"\n{args.model}  macro-F1 = {macro.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"   balanced acc = {bal.mean():.3f}")

    print(f"running {args.perms} label permutations...")
    p = permutation_test(X, y, args.model, macro.mean(), args.perms)
    print(f"permutation p = {p:.4f}")

    if args.out:
        row = pd.DataFrame([{"target": args.target, "model": args.model,
                             "macro_f1": macro.mean(), "ci_lo": lo, "ci_hi": hi,
                             "bal_acc": bal.mean(), "perm_p": p}])
        header = not pd.io.common.file_exists(args.out)
        row.to_csv(args.out, mode="a", header=header, index=False)
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
