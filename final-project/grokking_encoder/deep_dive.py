from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.nn import functional as F

from .config import ID_TO_LABEL
from .data import extract_top_label, load_mathnet_rows, load_npz_split
from .torch_train import get_device, iter_device_batches, load_device_split, load_model, make_mlm_batch


LABELS = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]


def softmax_predictions(run_dir: Path, checkpoint: Path, split: str, device_name: str, batch_size: int) -> dict[str, np.ndarray]:
    device = get_device(device_name)
    data = load_device_split(run_dir, split, device)
    model, _ = load_model(run_dir, device=device, checkpoint=checkpoint)
    model.eval()
    logits_out = []
    labels_out = []
    with torch.no_grad():
        for batch in iter_device_batches(data, batch_size=batch_size, shuffle=False):
            logits = model.classify(batch["input_ids"], batch["attention_mask"])
            logits_out.append(logits.detach().cpu())
            labels_out.append(batch["labels"].detach().cpu())
    logits = torch.cat(logits_out).numpy()
    labels = torch.cat(labels_out).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    return {"logits": logits, "probs": probs, "labels": labels, "preds": preds}


def masked_mlm_metrics(run_dir: Path, checkpoint: Path, split: str, device_name: str, batch_size: int) -> dict[str, float]:
    torch.manual_seed(123)
    device = get_device(device_name)
    data = load_device_split(run_dir, split, device)
    model, cfg = load_model(run_dir, device=device, checkpoint=checkpoint)
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iter_device_batches(data, batch_size=batch_size, shuffle=False):
            corrupted, positions, labels = make_mlm_batch(
                batch["input_ids"],
                batch["attention_mask"],
                cfg,
                mlm_prob=0.15,
                math_mask_prob=0.0,
                math_token_mask=None,
            )
            if labels.numel() == 0:
                continue
            hidden = model.encode(corrupted, batch["attention_mask"])
            logits = model.mlm_logits(hidden, positions)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            pred = logits.argmax(dim=-1)
            losses.append(float(loss.detach().cpu()))
            correct += int((pred == labels).sum().detach().cpu())
            total += int(labels.numel())
    loss = float(np.sum(losses) / max(1, total))
    return {
        "masked_tokens": float(total),
        "mlm_loss": loss,
        "mlm_accuracy": float(correct / max(1, total)),
        "mlm_perplexity": float(math.exp(loss)),
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0, "accuracy": np.nan, "confidence": np.nan})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - conf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": int(mask.sum()), "accuracy": acc, "confidence": conf})
    return {"ece": float(ece), "bins": rows}


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    preds = probs.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=np.arange(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=np.arange(len(LABELS)), zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=np.arange(len(LABELS)))
    true_probs = probs[np.arange(len(y_true)), y_true]
    losses = -np.log(np.clip(true_probs, 1e-12, 1.0))
    rows = []
    for i, label in enumerate(LABELS):
        mask = y_true == i
        class_probs = probs[:, i]
        auc = roc_auc_score(y_bin[:, i], class_probs)
        pr_auc = average_precision_score(y_bin[:, i], class_probs)
        brier = brier_score_loss(y_bin[:, i], class_probs)
        class_loss = float(losses[mask].mean())
        rows.append(
            {
                "category": label,
                "support": int(support[i]),
                "predicted_count": int((preds == i).sum()),
                "correct": int(cm[i, i]),
                "accuracy_recall": float(recall[i]),
                "precision": float(precision[i]),
                "f1": float(f1[i]),
                "roc_auc_ovr": float(auc),
                "pr_auc_ovr": float(pr_auc),
                "brier_ovr": float(brier),
                "mean_ce_loss": class_loss,
                "classification_perplexity": float(math.exp(class_loss)),
                "mean_true_probability": float(true_probs[mask].mean()),
                "mean_confidence": float(probs[mask].max(axis=1).mean()),
                "mean_entropy": float((-probs[mask] * np.log(np.clip(probs[mask], 1e-12, 1.0))).sum(axis=1).mean()),
                "mean_margin": float(np.sort(probs[mask], axis=1)[:, -1].mean() - np.sort(probs[mask], axis=1)[:, -2].mean()),
            }
        )
    class_df = pd.DataFrame(rows)
    overall = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(len(LABELS)))),
        "classification_perplexity": float(math.exp(log_loss(y_true, probs, labels=np.arange(len(LABELS))))),
        "macro_roc_auc_ovr": float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")),
        "weighted_roc_auc_ovr": float(roc_auc_score(y_bin, probs, average="weighted", multi_class="ovr")),
        "macro_pr_auc_ovr": float(np.mean([average_precision_score(y_bin[:, i], probs[:, i]) for i in range(len(LABELS))])),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, preds)),
        "cohen_kappa": float(cohen_kappa_score(y_true, preds)),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "mean_confidence_correct": float(probs.max(axis=1)[preds == y_true].mean()),
        "mean_confidence_wrong": float(probs.max(axis=1)[preds != y_true].mean()),
    }
    calib = expected_calibration_error(probs.max(axis=1), preds == y_true)
    overall["ece_15_bins"] = calib["ece"]
    calib_df = pd.DataFrame(calib["bins"])
    return overall, class_df, calib_df


def split_feature_stats(run_dir: Path, data_path: Path, split: str) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, float]]:
    texts, labels, _ = load_mathnet_rows(data_path)
    all_rows = json.loads(data_path.read_text())
    meta = []
    for row in all_rows:
        if extract_top_label(row.get("topics_flat")) is not None:
            meta.append(row)
    data = load_npz_split(run_dir, split)
    indices = data["indices"]
    y = data["labels"]
    word_re = re.compile(r"[A-Za-z]+|\\[A-Za-z]+|\d+|[$^_{}=+*/<>≤≥≡∠°π]+")
    geom_re = re.compile(
        r"\b(figure|diagram|shown|pictured|circle|triangle|quadrilateral|line|point|angle|side|radius|diameter|parallel|perpendicular|polygon|square|rectangle|tangent|incircle|circumcircle|bisector|altitude|orthocenter)\b",
        re.I,
    )
    comb_re = re.compile(
        r"\b(color|colour|graph|vertices|edges|tournament|game|choose|ways|arrangements|permutation|subset|set|sequence|domino|chess|knight|grid|cards|balls|boxes|committee|roads|cities|flights)\b",
        re.I,
    )
    algebra_re = re.compile(
        r"\b(polynomial|function|equation|roots|real|complex|solve|system|inequality|sequence|sum|product|matrix|derivative|quadratic|linear)\b",
        re.I,
    )
    nt_re = re.compile(
        r"\b(integer|divisible|divisor|prime|gcd|lcm|modulo|congruent|remainder|factor|multiple|natural|positive integers|diophantine)\b",
        re.I,
    )
    img_md_re = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img|https?://|datasets-server|cached-assets", re.I)
    rows = []
    for lid, label in enumerate(LABELS):
        ids = indices[y == lid]
        raw = [texts[int(i)] for i in ids]
        tokenized = [word_re.findall(t) for t in raw]
        enc_len = data["attention_mask"][y == lid].sum(axis=1)
        rows.append(
            {
                "category": label,
                "n": int(len(ids)),
                "median_chars": float(np.median([len(t) for t in raw])),
                "median_word_tokens": float(np.median([len(t) for t in tokenized])),
                "median_bpe_tokens": float(np.median(enc_len)),
                "truncated_rate": float((enc_len >= 256).mean()),
                "image_metadata_rate": float(np.mean([bool(meta[int(i)].get("images")) for i in ids])),
                "problem_markdown_image_rate": float(np.mean([bool(img_md_re.search(texts[int(i)])) for i in ids])),
                "geometry_cue_rate": float(np.mean([bool(geom_re.search(t)) for t in raw])),
                "combinatorics_cue_rate": float(np.mean([bool(comb_re.search(t)) for t in raw])),
                "algebra_cue_rate": float(np.mean([bool(algebra_re.search(t)) for t in raw])),
                "number_theory_cue_rate": float(np.mean([bool(nt_re.search(t)) for t in raw])),
            }
        )
    train_data = load_npz_split(run_dir, "train")
    train_texts = [texts[int(i)] for i in train_data["indices"]]
    train_labels = train_data["labels"]
    vec = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        min_df=5,
        binary=True,
        stop_words="english",
    )
    x = vec.fit_transform(train_texts)
    terms = np.array(vec.get_feature_names_out())
    top_words = {}
    for lid, label in enumerate(LABELS):
        mask = train_labels == lid
        c1 = np.asarray(x[mask].sum(axis=0)).ravel() + 1
        c0 = np.asarray(x[~mask].sum(axis=0)).ravel() + 1
        n1 = int(mask.sum()) + 2
        n0 = int((~mask).sum()) + 2
        score = np.log(c1 / (n1 - c1)) - np.log(c0 / (n0 - c0))
        top_words[label] = terms[np.argsort(score)[-20:][::-1]].tolist()
    vec2 = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    x_train = vec2.fit_transform(train_texts)
    x_test = vec2.transform([texts[int(i)] for i in data["indices"]])
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(x_train, train_labels)
    pred = clf.predict(x_test)
    tfidf = {"overall_accuracy": float(accuracy_score(y, pred))}
    for lid, label in enumerate(LABELS):
        mask = y == lid
        tfidf[f"{label}_accuracy"] = float(accuracy_score(y[mask], pred[mask]))
    return pd.DataFrame(rows), top_words, tfidf


def read_phase_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "val_macro_f1" in row]
    if not rows:
        return {}
    best = max(rows, key=lambda row: row["val_macro_f1"])
    final = rows[-1]
    return {
        "epochs": len(rows),
        "best": best,
        "final": final,
        "avg_tokens_per_s": float(np.mean([row.get("tokens_per_s", np.nan) for row in rows])),
    }


def save_prediction_tables(
    out_dir: Path,
    run_dir: Path,
    data_path: Path,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, str]:
    texts, _, _ = load_mathnet_rows(data_path)
    all_rows = json.loads(data_path.read_text())
    meta = []
    for row in all_rows:
        if extract_top_label(row.get("topics_flat")) is not None:
            meta.append(row)
    split = load_npz_split(run_dir, "test")
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    true_prob = probs[np.arange(len(y_true)), y_true]
    rows = []
    for row_id, idx in enumerate(split["indices"]):
        text = re.sub(r"\s+", " ", texts[int(idx)]).strip()
        rows.append(
            {
                "row": row_id,
                "dataset_index": int(idx),
                "id": meta[int(idx)].get("id", ""),
                "true": LABELS[int(y_true[row_id])],
                "pred": LABELS[int(preds[row_id])],
                "correct": bool(preds[row_id] == y_true[row_id]),
                "confidence": float(confidence[row_id]),
                "true_probability": float(true_prob[row_id]),
                "snippet": text[:260],
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_path = out_dir / "test_predictions.csv"
    err_path = out_dir / "high_confidence_errors.csv"
    pred_df.to_csv(pred_path, index=False)
    pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).head(120).to_csv(err_path, index=False)
    return {"predictions_csv": str(pred_path), "high_confidence_errors_csv": str(err_path)}


def plot_bar(df: pd.DataFrame, x: str, ys: list[str], path: Path, title: str, ylabel: str) -> None:
    ax = df.set_index(x)[ys].plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def build_plots(
    out_dir: Path,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_comparison: pd.DataFrame,
    run_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    plot_bar(
        model_comparison,
        "checkpoint",
        ["accuracy", "macro_f1"],
        out_dir / "model_comparison.png",
        "Checkpoint comparison on full test set",
        "score",
    )
    paths.append("model_comparison.png")
    plot_bar(
        class_df,
        "category",
        ["accuracy_recall", "precision", "f1", "roc_auc_ovr", "pr_auc_ovr"],
        out_dir / "per_category_scores.png",
        "Long-joint best per-category scores",
        "score",
    )
    paths.append("per_category_scores.png")
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=np.arange(len(LABELS)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Long-joint best confusion matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=170)
    plt.close()
    paths.append("confusion_matrix.png")
    plot_bar(
        class_df,
        "category",
        ["mean_ce_loss", "classification_perplexity", "mean_confidence", "mean_true_probability"],
        out_dir / "loss_ppl_confidence.png",
        "Per-category loss, classification perplexity, and confidence",
        "value",
    )
    paths.append("loss_ppl_confidence.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    correct = probs.argmax(axis=1) == y_true
    conf = probs.max(axis=1)
    ax.hist(conf[correct], bins=np.linspace(0, 1, 21), alpha=0.7, label="correct")
    ax.hist(conf[~correct], bins=np.linspace(0, 1, 21), alpha=0.7, label="wrong")
    ax.set_title("Prediction confidence distribution")
    ax.set_xlabel("max predicted probability")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_histogram.png", dpi=170)
    plt.close()
    paths.append("confidence_histogram.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    mids = (calib_df["bin_low"] + calib_df["bin_high"]) / 2
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":")
    ax.plot(mids, calib_df["accuracy"], marker="o", label="accuracy")
    ax.plot(mids, calib_df["confidence"], marker="o", label="confidence")
    ax.set_title("Calibration by confidence bin")
    ax.set_xlabel("confidence bin midpoint")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=170)
    plt.close()
    paths.append("calibration_curve.png")
    y_bin = label_binarize(y_true, classes=np.arange(len(LABELS)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{label} AUC {roc_auc_score(y_bin[:, i], probs[:, i]):.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_title("One-vs-rest ROC curves")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=170)
    plt.close()
    paths.append("roc_curves.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(LABELS):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(recall, precision, label=f"{label} AP {average_precision_score(y_bin[:, i], probs[:, i]):.3f}")
    ax.set_title("One-vs-rest precision-recall curves")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=170)
    plt.close()
    paths.append("pr_curves.png")
    plot_bar(
        feature_df,
        "category",
        ["geometry_cue_rate", "combinatorics_cue_rate", "algebra_cue_rate", "number_theory_cue_rate"],
        out_dir / "cue_rates.png",
        "Surface cue rates in the test split",
        "rate",
    )
    paths.append("cue_rates.png")
    plot_bar(
        feature_df,
        "category",
        ["median_bpe_tokens", "truncated_rate", "problem_markdown_image_rate"],
        out_dir / "text_length_and_markup.png",
        "Text length, truncation, and image markup",
        "value",
    )
    paths.append("text_length_and_markup.png")
    long_log = run_dir / "logs" / "long_joint.jsonl"
    if long_log.exists():
        rows = [json.loads(line) for line in long_log.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if "val_macro_f1" in row]
        df = pd.DataFrame(rows)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(df["epoch"], df["train_loss"], label="train loss")
        ax1.plot(df["epoch"], df["val_loss"], label="validation loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(df["epoch"], df["val_macro_f1"], color="tab:green", label="validation macro F1")
        ax2.set_ylabel("macro F1")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        ax1.set_title("Long-joint training curve")
        plt.tight_layout()
        plt.savefig(out_dir / "long_joint_training_curve.png", dpi=170)
        plt.close()
        paths.append("long_joint_training_curve.png")
    return paths


def markdown_table(df: pd.DataFrame, max_cols: int | None = None) -> str:
    if max_cols is not None:
        df = df.iloc[:, :max_cols]
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_notebook(out_dir: Path, metrics: dict[str, Any], plot_paths: list[str], top_words: dict[str, list[str]]) -> Path:
    per_class = pd.DataFrame(metrics["per_class"])
    model_comp = pd.DataFrame(metrics["model_comparison"])
    features = pd.DataFrame(metrics["feature_stats"])
    lines = [
        "**MathNet Category Deep Dive**",
        "",
        "This notebook summarizes the extra analysis requested after the long-joint run.",
        "",
        "**Overall Long-Joint Best Metrics**",
        "",
        f"- Accuracy: {metrics['overall']['accuracy']:.4f}",
        f"- Balanced accuracy: {metrics['overall']['balanced_accuracy']:.4f}",
        f"- Macro F1: {metrics['overall']['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['overall']['weighted_f1']:.4f}",
        f"- Log loss: {metrics['overall']['log_loss']:.4f}",
        f"- Classification perplexity: {metrics['overall']['classification_perplexity']:.4f}",
        f"- Macro ROC AUC: {metrics['overall']['macro_roc_auc_ovr']:.4f}",
        f"- Macro PR AUC: {metrics['overall']['macro_pr_auc_ovr']:.4f}",
        f"- ECE with 15 bins: {metrics['overall']['ece_15_bins']:.4f}",
        "",
        "**Checkpoint Comparison**",
        "",
        markdown_table(model_comp),
        "",
        "**Per-Class Metrics**",
        "",
        markdown_table(per_class),
        "",
        "**Feature Stats**",
        "",
        markdown_table(features),
        "",
        "**Distinctive Training Words**",
        "",
    ]
    for label, words in top_words.items():
        lines.append(f"- {label}: {', '.join(words[:15])}")
    lines.extend(
        [
        "",
        "**Interpretation**",
        "",
        "On the current long-joint best checkpoint, Combinatorics does not beat Algebra by accuracy or F1. Algebra recall is 0.8359 and Combinatorics recall is 0.8330. Algebra F1 is 0.8450 and Combinatorics F1 is 0.7971.",
        "",
        "**Why The Scores Look Discrepant**",
        "",
        "There are two separate effects. First, different metrics answer different questions. Per-category accuracy in this report is the same thing as recall for that true class. It asks: of the examples that truly belong to this category, how many were recovered. Precision asks: of the examples predicted as this category, how many were actually correct. F1 combines precision and recall.",
        "",
        "Combinatorics looks competitive on recall, but weak on precision. The model predicted Combinatorics 640 times, while the test set contains only 587 true Combinatorics examples. That means the model overuses the Combinatorics label. It correctly recovered 489 Combinatorics examples, but it also pulled in 151 false positives from other categories. This gives recall 0.8330 but precision only 0.7641, so F1 falls to 0.7971.",
        "",
        "Algebra has almost the same recall, 0.8359, but much better precision, 0.8544. The model predicted Algebra 769 times for 786 true Algebra examples. It missed some Algebra examples, especially ones with grids, colors, chairs, socks, and other discrete story language, but when it did predict Algebra it was more often right. That is why Algebra F1 is 0.8450, much higher than Combinatorics F1.",
        "",
        "Second, checkpoint choice changes small per-class recall comparisons. In the earlier short joint checkpoint, Combinatorics recall was 0.8382 and Algebra recall was 0.8295, so Combinatorics looked slightly better by recall. In the long-joint best checkpoint, Algebra recall is 0.8359 and Combinatorics recall is 0.8330, so Algebra is slightly better. The recall gap is tiny in both cases. The stable finding is not that Combinatorics is truly better than Algebra. The stable finding is that Geometry is much easier, and Number Theory is consistently harder.",
        "",
        "AUC adds another wrinkle. AUC measures ranking quality for one category against the rest across all possible thresholds, not the quality of the final argmax label. Number Theory has ROC AUC 0.9566, which is strong, even though its argmax recall is only 0.7726. That means the model often assigns useful relative probability to Number Theory, but the final top label is frequently stolen by Algebra or Combinatorics. For the actual classifier, the confusion matrix and F1 are more directly meaningful than AUC alone.",
        "",
        "Log loss and classification perplexity explain another difference. Number Theory has the worst mean cross-entropy loss, 0.6438, and worst classification perplexity, 1.9037. Combinatorics is next worst at 0.5323 loss and 1.7028 perplexity. Algebra is better at 0.4971 loss and 1.6439 perplexity. Geometry is far better at 0.1610 loss and 1.1747 perplexity. This tells us the model is not merely making more Number Theory mistakes. It is also less confident in the correct Number Theory label on average.",
        "",
        "Combinatorics can look strong on recall because many examples contain concrete discrete-object cues like colorings, graphs, games, grids, chess, roads, cities, flights, and arrangements. The tradeoff is precision. The model overpredicts Combinatorics for 64 Algebra examples, 26 Geometry examples, and 61 Number Theory examples.",
        "",
        "Number Theory does worse than Algebra because it is smaller, overlaps heavily with Algebra syntax, and often uses generic variables, equations, integer constraints, sequences, and polynomial-looking expressions. Its strong cues like prime, divisor, gcd, lcm, modulo, and congruent are helpful when present, but many Number Theory problems do not expose enough of those cues in the problem statement alone.",
            "",
            "Geometry remains the easiest category because its vocabulary is highly separable. The test split geometry-cue rate is 0.804, far higher than Algebra or Number Theory.",
            "",
            "**Graphs**",
            "",
        ]
    )
    for path in plot_paths:
        lines.append(f"![{path}]({path})")
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "\n",
                    "metrics = json.loads(Path(\"deep_dive_metrics.json\").read_text())\n",
                    "metrics[\"overall\"]\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = out_dir / "category_deep_dive.ipynb"
    path.write_text(json.dumps(nb, indent=2))
    return path


def main() -> int:
    run_dir = Path("runs/tiny_mathnet_grokking")
    data_path = Path("mathnet_dataset.json")
    checkpoint = run_dir / "checkpoints" / "long_joint_best.pt"
    out_dir = run_dir / "report" / "deep_dive"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = softmax_predictions(run_dir, checkpoint, "test", "mps", 512)
    y_true = pred["labels"]
    probs = pred["probs"]
    prediction_tables = save_prediction_tables(out_dir, run_dir, data_path, y_true, probs)
    overall, class_df, calib_df = classification_metrics(y_true, probs)
    feature_df, top_words, tfidf = split_feature_stats(run_dir, data_path, "test")
    mlm = masked_mlm_metrics(run_dir, checkpoint, "test", "mps", 512)
    eval_dir = run_dir / "eval"
    model_rows = []
    for name in ["supervised_best", "finetune_best", "joint_best", "long_joint_best"]:
        path = eval_dir / f"{name}_test.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        model_rows.append({"checkpoint": name, **data["metrics"]})
    model_df = pd.DataFrame(model_rows)
    class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    calib_df.to_csv(out_dir / "calibration_bins.csv", index=False)
    feature_df.to_csv(out_dir / "feature_stats.csv", index=False)
    model_df.to_csv(out_dir / "model_comparison.csv", index=False)
    plot_paths = build_plots(out_dir, y_true, probs, class_df, calib_df, feature_df, model_df, run_dir)
    metrics = {
        "overall": overall,
        "mlm_masked_token": mlm,
        "per_class": class_df.to_dict(orient="records"),
        "calibration_bins": calib_df.to_dict(orient="records"),
        "feature_stats": feature_df.to_dict(orient="records"),
        "top_words": top_words,
        "tfidf_baseline": tfidf,
        "model_comparison": model_df.to_dict(orient="records"),
        "long_joint_summary": read_phase_summary(run_dir / "logs" / "long_joint.jsonl"),
        "plots": plot_paths,
        "prediction_tables": prediction_tables,
    }
    (out_dir / "deep_dive_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    notebook = build_notebook(out_dir, metrics, plot_paths, top_words)
    print(json.dumps({"out_dir": str(out_dir), "notebook": str(notebook), "plots": plot_paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
