from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


class GlobalThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Binarize all features with one threshold and return integer 0/1."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr > self.threshold).astype(np.int8)


class GroupThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Use separate thresholds for word, char, and capital feature groups."""

    def __init__(
        self,
        word_threshold: float = 0.0,
        char_threshold: float = 0.0,
        capital_threshold: float = 1.0,
        feature_names: list[str] | None = None,
    ):
        self.word_threshold = word_threshold
        self.char_threshold = char_threshold
        self.capital_threshold = capital_threshold
        self.feature_names = feature_names

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            names = list(X.columns)
        elif self.feature_names is not None:
            names = list(self.feature_names)
        else:
            raise ValueError("GroupThresholdBinarizer requires DataFrame input or feature_names.")

        self.word_idx_ = [i for i, name in enumerate(names) if name.startswith("word_freq_")]
        self.char_idx_ = [i for i, name in enumerate(names) if name.startswith("char_freq_")]
        self.capital_idx_ = [i for i, name in enumerate(names) if name.startswith("capital_run_length_")]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        out = np.zeros_like(X_arr, dtype=np.int8)
        if self.word_idx_:
            out[:, self.word_idx_] = (X_arr[:, self.word_idx_] > self.word_threshold).astype(np.int8)
        if self.char_idx_:
            out[:, self.char_idx_] = (X_arr[:, self.char_idx_] > self.char_threshold).astype(np.int8)
        if self.capital_idx_:
            out[:, self.capital_idx_] = (X_arr[:, self.capital_idx_] > self.capital_threshold).astype(np.int8)
        return out


class QuantileBinarizer(BaseEstimator, TransformerMixin):
    """Binarize each feature using its own training-set quantile threshold."""

    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.thresholds_ = np.quantile(X_arr, self.quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr > self.thresholds_).astype(np.int8)


class TopKMutualInfoSelector(BaseEstimator, TransformerMixin):
    """Mutual-information selector that safely clamps k to available features."""

    def __init__(self, k: int | str = "all", random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_features = X_arr.shape[1]

        self.scores_ = mutual_info_classif(
            X_arr,
            y_arr,
            discrete_features=True,
            random_state=self.random_state,
        )

        if self.k == "all":
            self.k_effective_ = n_features
        else:
            self.k_effective_ = max(1, min(int(self.k), n_features))

        top_indices = np.argsort(self.scores_)[::-1][: self.k_effective_]
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        self.support_mask_[top_indices] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    feature_set: str
    scoring: str
    estimator: Pipeline
    param_grid: dict[str, list[Any]]


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_experiment(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        cv=5,
        scoring=spec.scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    y_prob = best_estimator.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    return {
        "search": search,
        "best_estimator": best_estimator,
        "best_params": search.best_params_,
        "cv_best_score": search.best_score_,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


def feature_names_after_transforms(estimator: Pipeline, input_feature_names: list[str]) -> list[str]:
    names = np.array(input_feature_names)
    for step_name, step in estimator.named_steps.items():
        if step_name == "model":
            break
        if hasattr(step, "get_support"):
            names = names[step.get_support()]
    return names.tolist()


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_comparison(y_true, prob_a, prob_b, name_a: str, name_b: str, output_path: Path):
    fpr_a, tpr_a, _ = roc_curve(y_true, prob_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_b)
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_a, tpr_a, label=f"{name_a} (AUC={auc_a:.3f})")
    ax.plot(fpr_b, tpr_b, label=f"{name_b} (AUC={auc_b:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_precision_recall_comparison(y_true, prob_a, prob_b, name_a: str, name_b: str, output_path: Path):
    precision_a, recall_a, _ = precision_recall_curve(y_true, prob_a)
    precision_b, recall_b, _ = precision_recall_curve(y_true, prob_b)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_a, precision_a, label=name_a)
    ax.plot(recall_b, precision_b, label=name_b)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metric_comparison_plot(results_df: pd.DataFrame, metric: str, output_path: Path):
    plot_df = results_df.sort_values(metric, ascending=False)
    values = plot_df[metric].to_numpy(dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    spread = max(vmax - vmin, 1e-4)
    pad = max(spread * 0.35, 0.003)
    y_low = max(0.0, vmin - pad)
    y_high = min(1.0, vmax + pad)

    # Keep the chart zoomed in so small metric differences are visible.
    if y_high - y_low < 0.01:
        center = (y_high + y_low) / 2.0
        y_low = max(0.0, center - 0.005)
        y_high = min(1.0, center + 0.005)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_df["name"], plot_df[metric])
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Experiment Comparison: {metric.upper()}")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_experiments(general_features: list[str]) -> list[ExperimentSpec]:
    alpha_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    global_threshold_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

    return [
        ExperimentSpec(
            name="global_full_f1",
            description="Apples-to-apples with all features: global threshold + F1 tuning.",
            feature_set="all",
            scoring="f1",
            estimator=Pipeline(
                [
                    ("binarizer", GlobalThresholdBinarizer()),
                    ("model", BernoulliNB(binarize=None)),
                ]
            ),
            param_grid={
                "binarizer__threshold": global_threshold_grid,
                "model__alpha": alpha_grid,
            },
        ),
        ExperimentSpec(
            name="global_no_domain_f1",
            description="Remove HP-specific features; tune global threshold + alpha on F1.",
            feature_set="general",
            scoring="f1",
            estimator=Pipeline(
                [
                    ("binarizer", GlobalThresholdBinarizer()),
                    ("model", BernoulliNB(binarize=None)),
                ]
            ),
            param_grid={
                "binarizer__threshold": global_threshold_grid,
                "model__alpha": alpha_grid,
            },
        ),
        ExperimentSpec(
            name="global_no_domain_f1_var_mi",
            description="Global threshold + variance filter + MI selector.",
            feature_set="general",
            scoring="f1",
            estimator=Pipeline(
                [
                    ("binarizer", GlobalThresholdBinarizer()),
                    ("variance", VarianceThreshold()),
                    ("selector", TopKMutualInfoSelector()),
                    ("model", BernoulliNB(binarize=None)),
                ]
            ),
            param_grid={
                "binarizer__threshold": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "variance__threshold": [0.0, 0.001],
                "selector__k": [20, 30, "all"],
                "model__alpha": [0.01, 0.1, 0.5, 1.0],
            },
        ),
        ExperimentSpec(
            name="grouped_no_domain_f1_var_mi",
            description="Separate thresholds for word/char/capital groups.",
            feature_set="general",
            scoring="f1",
            estimator=Pipeline(
                [
                    ("binarizer", GroupThresholdBinarizer(feature_names=general_features)),
                    ("variance", VarianceThreshold()),
                    ("selector", TopKMutualInfoSelector()),
                    ("model", BernoulliNB(binarize=None)),
                ]
            ),
            param_grid={
                "binarizer__word_threshold": [0.0, 0.1, 0.5],
                "binarizer__char_threshold": [0.0, 0.1],
                "binarizer__capital_threshold": [1.0, 5.0, 25.0],
                "variance__threshold": [0.001],
                "selector__k": [20, "all"],
                "model__alpha": [0.01, 0.1, 0.5, 1.0],
            },
        ),
        ExperimentSpec(
            name="quantile_no_domain_f1_var_mi",
            description="Per-feature quantile thresholds (feature-scale aware).",
            feature_set="general",
            scoring="f1",
            estimator=Pipeline(
                [
                    ("binarizer", QuantileBinarizer()),
                    ("variance", VarianceThreshold()),
                    ("selector", TopKMutualInfoSelector()),
                    ("model", BernoulliNB(binarize=None)),
                ]
            ),
            param_grid={
                "binarizer__quantile": [0.25, 0.4, 0.5, 0.6, 0.75],
                "variance__threshold": [0.0, 0.001],
                "selector__k": [20, "all"],
                "model__alpha": [0.01, 0.1, 0.5, 1.0],
            },
        ),
    ]


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spambase" / "spambase.data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATE_LABEL = datetime.now().strftime("%Y-%m-%d")

ALL_FEATURE_COLUMNS = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_semicolon",
    "char_freq_paren",
    "char_freq_bracket",
    "char_freq_exclaim",
    "char_freq_dollar",
    "char_freq_hash",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
DOMAIN_SPECIFIC_FEATURES = [
    "word_freq_george",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_650",
    "word_freq_857",
]
GENERAL_FEATURE_COLUMNS = [f for f in ALL_FEATURE_COLUMNS if f not in DOMAIN_SPECIFIC_FEATURES]

columns = ALL_FEATURE_COLUMNS + ["spam"]
df = pd.read_csv(DATA_PATH, header=None, names=columns)

X_all = df[ALL_FEATURE_COLUMNS]
y = df["spam"]
X_general = X_all.drop(columns=DOMAIN_SPECIFIC_FEATURES)

X_all_train, X_all_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=42, stratify=y
)
X_general_train = X_all_train[GENERAL_FEATURE_COLUMNS]
X_general_test = X_all_test[GENERAL_FEATURE_COLUMNS]

feature_sets = {
    "all": (X_all_train, X_all_test, ALL_FEATURE_COLUMNS),
    "general": (X_general_train, X_general_test, GENERAL_FEATURE_COLUMNS),
}

experiments = build_experiments(GENERAL_FEATURE_COLUMNS)
all_outcomes = {}
summary_rows = []

print("Running experiments...")
for i, spec in enumerate(experiments, start=1):
    X_train, X_test, input_features = feature_sets[spec.feature_set]
    print(f"[{i}/{len(experiments)}] {spec.name}: {spec.description}")
    outcome = run_experiment(spec, X_train, y_train, X_test, y_test)
    selected_features = feature_names_after_transforms(outcome["best_estimator"], input_features)

    all_outcomes[spec.name] = {
        "spec": spec,
        "outcome": outcome,
        "selected_features": selected_features,
        "input_features": input_features,
    }

    metrics = outcome["metrics"]
    summary_rows.append(
        {
            "name": spec.name,
            "description": spec.description,
            "feature_set": spec.feature_set,
            "scoring": spec.scoring,
            "cv_best_score": outcome["cv_best_score"],
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "selected_feature_count": len(selected_features),
            "best_params": json.dumps(outcome["best_params"], sort_keys=True),
        }
    )

results_df = pd.DataFrame(summary_rows)
results_df = results_df.sort_values(["test_f1", "test_roc_auc"], ascending=False).reset_index(drop=True)
results_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

print()
print("=== Experiment Summary (sorted by test F1) ===")
print(
    results_df[
        [
            "name",
            "cv_best_score",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "selected_feature_count",
        ]
    ].round(4).to_string(index=False)
)
print()

print("=== CV Score vs Test Score Check ===")
for _, row in results_df.iterrows():
    print(
        f"{row['name']}: CV({row['scoring']})={row['cv_best_score']:.4f}, "
        f"Test F1={row['test_f1']:.4f}, Test Acc={row['test_accuracy']:.4f}"
    )

best_name = results_df.iloc[0]["name"]
best_bundle = all_outcomes[best_name]
best_outcome = best_bundle["outcome"]

generalizable_df = results_df[results_df["feature_set"] == "general"].reset_index(drop=True)
best_generalizable_name = generalizable_df.iloc[0]["name"]
best_generalizable_bundle = all_outcomes[best_generalizable_name]
best_generalizable_outcome = best_generalizable_bundle["outcome"]

comparison_name = best_generalizable_name
if comparison_name == best_name and len(results_df) > 1:
    comparison_name = results_df.iloc[1]["name"]
comparison_outcome = all_outcomes[comparison_name]["outcome"]

print()
print(f"Best model by test F1: {best_name}")
print("Best params:", best_outcome["best_params"])
print("Best CV score:", round(best_outcome["cv_best_score"], 4))
print()
print(f"Best generalizable model (domain-specific features removed): {best_generalizable_name}")
print("Best params:", best_generalizable_outcome["best_params"])
print("Best CV score:", round(best_generalizable_outcome["cv_best_score"], 4))

# Model interpretability for best model.
best_estimator = best_outcome["best_estimator"]
best_selected_features = best_bundle["selected_features"]
model_step = best_estimator.named_steps["model"]
classes = list(model_step.classes_)
spam_idx = classes.index(1)
non_spam_idx = classes.index(0)

feature_analysis = pd.DataFrame(
    {
        "feature": best_selected_features,
        "log_prob_spam": model_step.feature_log_prob_[spam_idx],
        "log_prob_non_spam": model_step.feature_log_prob_[non_spam_idx],
    }
)
feature_analysis["log_odds_spam_minus_non_spam"] = (
    feature_analysis["log_prob_spam"] - feature_analysis["log_prob_non_spam"]
)
feature_analysis = feature_analysis.sort_values("log_odds_spam_minus_non_spam", ascending=False)
feature_analysis.to_csv(RESULTS_DIR / "best_model_feature_log_odds.csv", index=False)

print()
print("=== Top 10 Spam-Indicative Features (Best Model) ===")
print(feature_analysis.head(10)[["feature", "log_odds_spam_minus_non_spam"]].round(4).to_string(index=False))
print()
print("=== Top 10 Non-Spam-Indicative Features (Best Model) ===")
print(
    feature_analysis.tail(10)
    .iloc[::-1][["feature", "log_odds_spam_minus_non_spam"]]
    .round(4)
    .to_string(index=False)
)

# Save dated images.
best_confusion_path = RESULTS_DIR / f"best_model_confusion_matrix_{DATE_LABEL}.png"
comparison_confusion_path = RESULTS_DIR / f"comparison_model_confusion_matrix_{DATE_LABEL}.png"
roc_comparison_path = RESULTS_DIR / f"model_roc_curve_comparison_{DATE_LABEL}.png"
pr_comparison_path = RESULTS_DIR / f"model_precision_recall_curve_comparison_{DATE_LABEL}.png"
f1_comparison_path = RESULTS_DIR / f"experiment_test_f1_comparison_{DATE_LABEL}.png"
accuracy_comparison_path = RESULTS_DIR / f"experiment_test_accuracy_comparison_{DATE_LABEL}.png"

save_confusion_matrix(
    y_test,
    best_outcome["y_pred"],
    f"{best_name} Confusion Matrix",
    best_confusion_path,
)
save_confusion_matrix(
    y_test,
    comparison_outcome["y_pred"],
    f"{comparison_name} Confusion Matrix",
    comparison_confusion_path,
)
save_roc_comparison(
    y_test,
    best_outcome["y_prob"],
    comparison_outcome["y_prob"],
    best_name,
    comparison_name,
    roc_comparison_path,
)
save_precision_recall_comparison(
    y_test,
    best_outcome["y_prob"],
    comparison_outcome["y_prob"],
    best_name,
    comparison_name,
    pr_comparison_path,
)
save_metric_comparison_plot(results_df, "test_f1", f1_comparison_path)
save_metric_comparison_plot(results_df, "test_accuracy", accuracy_comparison_path)

print()
print("Saved images:")
print(best_confusion_path.name)
print(comparison_confusion_path.name)
print(roc_comparison_path.name)
print(pr_comparison_path.name)
print(f1_comparison_path.name)
print(accuracy_comparison_path.name)
