# determiner-comprhension-ml

Code for the paper *Classifying Determiner-Comprehension Subgroups and Clinical
Status in Children: A Machine Learning Approach*.

It predicts a data-driven determiner-comprehension grouping and clinical/language
status from three sources: mouse-clicking behavior in a language game, eye
tracking, and standardized tests, using leakage-controlled nested
cross-validation with permutation testing.

## Install

```
pip install -r requirements.txt
```

## Quick start

The `examples/` folder ships synthetic data so everything runs out of the box.

```
# train and evaluate one model on a feature table
python train.py --data examples/demo_features.csv --target group --model rf

# rank feature importance with SHAP
python shap_importance.py --data examples/demo_features.csv --target group

# run the tests
pytest
```

A feature table is one row per child: an `id` column, the target column, and
numeric feature columns.

## Layout

- `train.py` - nested CV, in-fold feature selection, permutation test (main entry point)
- `stats.py` - Benjamini-Hochberg FDR and the matched single-vs-combined test
- `shap_importance.py` - SHAP feature ranking
- `pipeline/` - data ingest and feature extraction from the raw recordings
- `examples/` - synthetic feature table and a sample run
- `tests/` - smoke tests

## Data

The real data is available from the authors on reasonable request under a data
use agreement (see the paper's Data Availability statement). The scripts
reproduce the reported results from the de-identified feature tables.
