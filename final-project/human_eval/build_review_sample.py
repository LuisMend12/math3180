"""Build a blind human-review sample for the MathNet topic classifier.

Reproduces the notebook's label mapping and 80/20 stratified split
(random_state=42), refits the best model (TF-IDF + Ridge), predicts on the
validation set, and emits a review packet that human reviewers can label
without seeing either the dataset label or the model's prediction.

Two strata are drawn and then shuffled together into one sheet:

  random  - a uniform random sample of validation items. Used to ESTIMATE
            accuracy against human labels. Unbiased, so the estimate and its
            confidence interval come from this stratum alone.
  error   - an oversample of items the model got wrong w.r.t. the dataset
            label. Used to CHARACTERIZE failures, not to estimate accuracy.

Reviewers must not be able to tell the strata apart, which is why the sheet is
shuffled and the stratum is recorded only in the answer key.

Usage:
    python build_review_sample.py --n-random 150 --n-error 60 --reviewers 3
"""

import argparse
import ast
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATASET = PROJECT / "mathnet_dataset.json"
OUT = HERE / "sample"

SEED = 42  # matches the notebook split

# Identical to CATEGORY_MAP in mathnet_cnn.ipynb.
CATEGORY_MAP = {
    "Algebra": "Algebra",
    "Number Theory": "Number Theory",
    "Geometry": "Geometry",
    "Trigonometry": "Geometry",
    "Discrete Mathematics": "Combinatorics",
    "Combinatorics": "Combinatorics",
}

CLASSES = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]


def parse_topics(topics_raw):
    """Return the raw topic strings for a row as a list."""
    try:
        if isinstance(topics_raw, list):
            return topics_raw
        return ast.literal_eval(str(topics_raw))
    except Exception:
        return []


def top_level(topic):
    """'Discrete Mathematics > Combinatorics > Coloring' -> 'Discrete Mathematics'."""
    return topic.split(">")[0].strip()


def extract_top_label(topics_raw):
    """The notebook's label rule: map the FIRST topic's top level, else None."""
    topics = parse_topics(topics_raw)
    if topics:
        return CATEGORY_MAP.get(top_level(topics[0]))
    return None


def mapped_class_set(topics_raw):
    """Every one of the four classes this row's topic list touches, in order."""
    seen = []
    for topic in parse_topics(topics_raw):
        cls = CATEGORY_MAP.get(top_level(topic))
        if cls and cls not in seen:
            seen.append(cls)
    return seen


def load_validation_frame():
    """Rebuild the notebook's dataframe, split, and model; return the val set."""
    print(f"Loading {DATASET.name} ...")
    df = pd.DataFrame(json.loads(DATASET.read_text(encoding="utf-8")))
    df = df.dropna(subset=["problem_markdown", "topics_flat"]).reset_index(drop=True)

    df["label"] = df["topics_flat"].apply(extract_top_label)
    df["all_classes"] = df["topics_flat"].apply(mapped_class_set)
    df = df[df["label"].notna()].reset_index(drop=True)
    print(f"Rows after label filtering: {len(df):,}")

    train_idx, val_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    train, val = df.loc[train_idx], df.loc[val_idx].copy()
    print(f"Train: {len(train):,} | Validation: {len(val):,}")

    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    x_train = tfidf.fit_transform(train["problem_markdown"].astype(str))
    x_val = tfidf.transform(val["problem_markdown"].astype(str))

    clf = RidgeClassifier(alpha=1.0).fit(x_train, train["label"])
    val["model_pred"] = clf.predict(x_val)
    val["model_correct"] = val["model_pred"] == val["label"]

    acc = accuracy_score(val["label"], val["model_pred"])
    print(f"TF-IDF + Ridge validation accuracy: {acc:.4f}")
    return val, acc


def report_multilabel_overlap(val):
    """How often the dataset's single label discards a topic the row also has.

    This is the measurable part of the label-noise problem: when a problem is
    tagged with several topics, the notebook keeps only the first, so a model
    naming one of the discarded topics is scored wrong even when its answer is
    defensible. Human review is what separates those from real errors.
    """
    multi = val["all_classes"].apply(len) > 1
    errors = ~val["model_correct"]
    pred_in_topics = val.apply(
        lambda r: r["model_pred"] in r["all_classes"], axis=1
    )
    excusable = errors & pred_in_topics

    print("\n--- Label ambiguity in the validation set ---")
    print(f"Items touching >1 of the four classes: {multi.sum():,} ({multi.mean():.1%})")
    print(f"Model errors vs dataset label:          {errors.sum():,} ({errors.mean():.1%})")
    print(
        f"  ...of which the prediction IS in the row's own topic list: "
        f"{excusable.sum():,} ({excusable.sum() / max(errors.sum(), 1):.1%} of errors)"
    )
    print(
        f"Accuracy if any listed topic counts as correct: "
        f"{(val['model_correct'] | pred_in_topics).mean():.4f}"
    )
    return {
        "multi_topic_rate": float(multi.mean()),
        "error_rate": float(errors.mean()),
        "errors_matching_a_listed_topic": int(excusable.sum()),
        "lenient_accuracy": float((val["model_correct"] | pred_in_topics).mean()),
    }


def build_sheets(val, n_random, n_error, reviewers):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    random_stratum = val.sample(n=n_random, random_state=SEED)
    remaining = val.drop(random_stratum.index)
    error_pool = remaining[~remaining["model_correct"]]
    error_stratum = error_pool.sample(
        n=min(n_error, len(error_pool)), random_state=SEED
    )

    sample = pd.concat(
        [random_stratum.assign(stratum="random"), error_stratum.assign(stratum="error")]
    )
    order = list(range(len(sample)))
    rng.shuffle(order)
    sample = sample.iloc[order].reset_index(drop=True)
    sample.insert(0, "item_id", [f"R{i:03d}" for i in range(1, len(sample) + 1)])

    # Answer key: never given to reviewers until their sheets are submitted.
    key = sample[
        ["item_id", "id", "stratum", "label", "model_pred", "model_correct"]
    ].copy()
    key["all_listed_classes"] = sample["all_classes"].apply("|".join)
    key = key.rename(columns={"id": "source_id", "label": "dataset_label"})
    key.to_csv(OUT / "answer_key.csv", index=False)

    # Blind sheet: problem text only, plus empty columns for the reviewer.
    for r in range(1, reviewers + 1):
        blind = pd.DataFrame(
            {
                "item_id": sample["item_id"],
                "problem_text": sample["problem_markdown"],
                "primary_topic": "",
                "secondary_topic": "",
                "confidence": "",
                "notes": "",
            }
        )
        blind.to_csv(OUT / f"review_sheet_reviewer{r}.csv", index=False, encoding="utf-8")

    # Readable packet, because long LaTeX in a spreadsheet cell is unusable.
    lines = [
        "# MathNet Human Review Packet",
        "",
        f"{len(sample)} problems. For each, read the statement and record in your CSV:",
        "",
        "- `primary_topic`: one of Algebra, Combinatorics, Geometry, Number Theory",
        "- `secondary_topic`: another class if the problem genuinely belongs to two, else blank",
        "- `confidence`: high / medium / low",
        "- `notes`: optional, e.g. why it was hard to place",
        "",
        "Do not discuss items with the other reviewers until all sheets are in.",
        "",
        "---",
        "",
    ]
    for _, row in sample.iterrows():
        lines += [f"## {row['item_id']}", "", str(row["problem_markdown"]).strip(), "", "---", ""]
    (OUT / "review_packet.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {len(sample)} items to {OUT}")
    print(f"  random stratum: {len(random_stratum)} | error stratum: {len(error_stratum)}")
    print(f"  {reviewers} blind sheets + answer_key.csv + review_packet.md")
    return sample


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-random", type=int, default=150, help="items for the estimation stratum")
    ap.add_argument("--n-error", type=int, default=60, help="extra model errors for the taxonomy")
    ap.add_argument("--reviewers", type=int, default=3, help="blind sheets to emit")
    args = ap.parse_args()

    val, acc = load_validation_frame()
    stats = report_multilabel_overlap(val)
    sample = build_sheets(val, args.n_random, args.n_error, args.reviewers)

    stats.update(
        {
            "validation_accuracy": acc,
            "validation_size": int(len(val)),
            "n_random": int(args.n_random),
            "n_error": int(len(sample) - args.n_random),
        }
    )
    (OUT / "sample_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
