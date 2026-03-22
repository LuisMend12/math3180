# Spam Email Classification with Bernoulli Naive Bayes

## Mathematical Background

This project models spam detection with Bernoulli Naive Bayes after converting features to binary indicators.

Bayes rule:

$$
P(Y=k \mid X)=\frac{P(X \mid Y=k)P(Y=k)}{P(X)}
$$

Naive conditional independence:

$$
P(X \mid Y=k)=\prod_{j=1}^{p} P(X_j \mid Y=k)
$$

Bernoulli likelihood:

$$
P(X_j=1 \mid Y=k)=\theta_{jk}, \quad P(X_j=0 \mid Y=k)=1-\theta_{jk}
$$

$$
P(X \mid Y=k)=\prod_{j=1}^{p} \theta_{jk}^{x_j}(1-\theta_{jk})^{1-x_j}
$$

Prediction rule:

$$
\hat y=\arg\max_k\left[\log P(Y=k)+\sum_{j=1}^{p}x_j\log\theta_{jk}+(1-x_j)\log(1-\theta_{jk})\right]
$$

## Dataset

Source: UCI Machine Learning Repository (Spambase)

- File: `midterm-project/spambase/spambase.data`
- Rows: 4601
- Features: 57 predictors + 1 label
- Class distribution: 1813 spam (39.4%), 2788 non-spam (60.6%)

## Perfect Implementation (Current)

`main.py` now runs a full ablation and threshold-variation suite with leakage-safe pipelines:

- Global-threshold tuning on all features (F1 optimization)
- Global-threshold tuning after dropping domain-specific HP features
- Global threshold + variance filter + mutual-information selection
- Grouped thresholds (word/char/capital)
- Per-feature quantile thresholds

Key improvements implemented:

- Binarization done with custom transformers that output integer 0/1 features
- Cross-validation score explicitly reported (`best_score_`)
- Apples-to-apples ablations included
- Multiple threshold strategy tests included
- Feature interpretability from `feature_log_prob_` included

## Latest Run Summary

From the most recent run of `python midterm-project/main.py`:

- Best model: `global_full_f1`
- Best params: `threshold=0.1`, `alpha=0.1`
- Best test accuracy: `0.9037`
- Best test F1: `0.8735`

## Important Outputs

Generated under `midterm-project/results/`:

- `experiment_summary.csv` (all variants and CV/test metrics)
- `best_model_confusion_matrix_<YYYY-MM-DD>.png`
- `comparison_model_confusion_matrix_<YYYY-MM-DD>.png`
- `model_roc_curve_comparison_<YYYY-MM-DD>.png`
- `model_precision_recall_curve_comparison_<YYYY-MM-DD>.png`
- `best_model_feature_log_odds.csv`
- `experiment_test_f1_comparison_<YYYY-MM-DD>.png`
- `experiment_test_accuracy_comparison_<YYYY-MM-DD>.png`

## How To Run

### Script

```bash
cd /Users/tanushv/Downloads/math3180-main
source .venv/bin/activate
python midterm-project/main.py
```

### Notebook

```bash
cd /Users/tanushv/Downloads/math3180-main
source .venv/bin/activate
jupyter lab
```

Then open `midterm-project/First_Project_Assignment.ipynb`.

## References

- UCI Machine Learning Repository: Spambase
- scikit-learn documentation for `BernoulliNB`
