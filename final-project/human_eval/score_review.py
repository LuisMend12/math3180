"""Score completed human-review sheets against the model and the dataset labels.

Reads the blind sheets produced by build_review_sample.py once reviewers have
filled them in, then reports:

  1. Inter-annotator agreement (pairwise Cohen's kappa, overall Fleiss' kappa).
     This establishes the human ceiling: the model cannot meaningfully be held
     to a standard the reviewers themselves do not reach.
  2. Model accuracy against the human consensus label, with a Wilson interval,
     computed on the random stratum only so the estimate stays unbiased.
  3. Agreement between the human consensus and the dataset label, which
     measures label quality rather than model quality.
  4. An error taxonomy over the error stratum, splitting apparent mistakes into
     genuine model errors, bad dataset labels, and genuinely ambiguous items.

Usage:
    python score_review.py                 # reads sample/, writes results/
    python score_review.py --sample-dir sample --out-dir results
"""

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = Path(__file__).resolve().parent
CLASSES = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval; behaves sensibly at small n, unlike normal approx."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def fleiss_kappa(table):
    """Fleiss' kappa from an items x categories count matrix."""
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)
    if n_items == 0 or (n_raters < 2).any():
        return float("nan")
    n = n_raters[0]
    if not (n_raters == n).all():
        # Unequal rater counts: restrict to the items everyone rated.
        keep = n_raters == n_raters.max()
        table, n = table[keep], n_raters.max()
        n_items = table.shape[0]
    p_i = ((table**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = table.sum(axis=0) / (n_items * n)
    p_e = (p_j**2).sum()
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def load_sheets(sample_dir):
    key = pd.read_csv(sample_dir / "answer_key.csv")
    sheets = {}
    for path in sorted(sample_dir.glob("review_sheet_reviewer*.csv")):
        df = pd.read_csv(path).set_index("item_id")
        name = path.stem.replace("review_sheet_", "")
        df["primary_topic"] = df["primary_topic"].astype(str).str.strip()
        unfilled = (~df["primary_topic"].isin(CLASSES)).sum()
        if unfilled:
            print(f"  WARNING: {name} has {unfilled} unlabeled/invalid rows; they are excluded.")
        sheets[name] = df
    if not sheets:
        raise SystemExit(f"No filled review sheets found in {sample_dir}")
    print(f"Loaded {len(sheets)} reviewer sheets covering {len(key)} items.")
    return key, sheets


def build_consensus(key, sheets):
    """Majority primary label per item; ties are recorded as no-consensus."""
    rows = []
    for item_id in key["item_id"]:
        votes = []
        secondaries = set()
        for df in sheets.values():
            if item_id not in df.index:
                continue
            primary = df.at[item_id, "primary_topic"]
            if primary in CLASSES:
                votes.append(primary)
            sec = df.at[item_id, "secondary_topic"]
            if isinstance(sec, str) and sec.strip() in CLASSES:
                secondaries.add(sec.strip())
        counts = Counter(votes)
        if not counts:
            consensus, unanimous, tied = None, False, False
        else:
            top = counts.most_common()
            tied = len(top) > 1 and top[0][1] == top[1][1]
            consensus = None if tied else top[0][0]
            unanimous = len(counts) == 1 and len(votes) == len(sheets)
        rows.append(
            {
                "item_id": item_id,
                "consensus": consensus,
                "unanimous": unanimous,
                "tied": tied,
                "n_votes": len(votes),
                "accepted_set": "|".join(sorted(set(votes) | secondaries)),
            }
        )
    return key.merge(pd.DataFrame(rows), on="item_id")


def agreement_section(key, sheets, out):
    out.append("## 1. Inter-annotator agreement\n")
    names = list(sheets)
    pairwise = []
    for a, b in itertools.combinations(names, 2):
        common = [
            i
            for i in key["item_id"]
            if sheets[a].get("primary_topic", {}).get(i) in CLASSES
            and sheets[b].get("primary_topic", {}).get(i) in CLASSES
        ]
        if not common:
            continue
        k = cohen_kappa_score(
            [sheets[a].at[i, "primary_topic"] for i in common],
            [sheets[b].at[i, "primary_topic"] for i in common],
            labels=CLASSES,
        )
        raw = sum(
            sheets[a].at[i, "primary_topic"] == sheets[b].at[i, "primary_topic"]
            for i in common
        ) / len(common)
        pairwise.append((a, b, len(common), raw, k))

    out.append("| Pair | Items | Raw agreement | Cohen's kappa |")
    out.append("|---|---:|---:|---:|")
    for a, b, n, raw, k in pairwise:
        out.append(f"| {a} vs {b} | {n} | {fmt_pct(raw)} | {k:.3f} |")

    counts = []
    for item_id in key["item_id"]:
        row = [0] * len(CLASSES)
        for df in sheets.values():
            p = df.at[item_id, "primary_topic"] if item_id in df.index else None
            if p in CLASSES:
                row[CLASSES.index(p)] += 1
        counts.append(row)
    fk = fleiss_kappa(pd.DataFrame(counts).to_numpy())
    out.append(f"\n**Fleiss' kappa (all reviewers): {fk:.3f}**\n")
    out.append(
        "Interpretation: kappa above 0.80 is near-perfect agreement, 0.60-0.80 "
        "substantial, 0.40-0.60 moderate. Whatever the reviewers cannot agree on "
        "is an upper bound on what any model can be expected to get right.\n"
    )
    return {"fleiss_kappa": None if math.isnan(fk) else fk,
            "pairwise": [{"pair": f"{a}|{b}", "n": n, "raw": raw, "kappa": k}
                         for a, b, n, raw, k in pairwise]}


def accuracy_section(scored, out):
    """Estimates come from the random stratum only - the error stratum is biased."""
    est = scored[(scored["stratum"] == "random") & scored["consensus"].notna()]
    n = len(est)
    out.append("## 2. Model accuracy on the random stratum\n")
    out.append(f"Estimation sample: {n} items (random stratum, consensus reached).\n")

    strict = (est["model_pred"] == est["consensus"]).sum()
    lenient = est.apply(
        lambda r: r["model_pred"] in str(r["accepted_set"]).split("|"), axis=1
    ).sum()
    vs_dataset = (est["model_pred"] == est["dataset_label"]).sum()
    human_vs_dataset = (est["consensus"] == est["dataset_label"]).sum()

    rows = [
        ("Model vs dataset label (reported metric)", vs_dataset),
        ("Model vs human consensus", strict),
        ("Model vs any topic a reviewer accepted", lenient),
        ("Human consensus vs dataset label (label quality)", human_vs_dataset),
    ]
    out.append("| Comparison | Correct | Rate | 95% Wilson CI |")
    out.append("|---|---:|---:|---:|")
    for name, k in rows:
        lo, hi = wilson_interval(k, n)
        out.append(f"| {name} | {k}/{n} | {fmt_pct(k / n)} | {fmt_pct(lo)} - {fmt_pct(hi)} |")
    out.append("")
    out.append(
        "The last row is the one the reported accuracy has to be read against: it "
        "is how often the dataset's own label survives human scrutiny. A model "
        "scored against imperfect labels cannot exceed that ceiling by much, so "
        "the gap between rows 1 and 2 is the part of the error rate that belongs "
        "to the labels rather than to the model.\n"
    )

    cm = confusion_matrix(est["consensus"], est["model_pred"], labels=CLASSES)
    out.append("### Confusion matrix vs human consensus (rows = human, cols = model)\n")
    out.append("| | " + " | ".join(CLASSES) + " |")
    out.append("|---|" + "---:|" * len(CLASSES))
    for cls, row in zip(CLASSES, cm):
        out.append(f"| **{cls}** | " + " | ".join(str(v) for v in row) + " |")
    out.append("")
    return {
        "n_estimation": int(n),
        "model_vs_dataset": float(vs_dataset / n) if n else None,
        "model_vs_consensus": float(strict / n) if n else None,
        "model_vs_accepted_set": float(lenient / n) if n else None,
        "consensus_vs_dataset": float(human_vs_dataset / n) if n else None,
    }


def taxonomy_section(scored, out, out_dir):
    """Why does the model look wrong? Three very different answers."""
    errs = scored[(scored["model_pred"] != scored["dataset_label"])]
    out.append("## 3. Error taxonomy\n")
    out.append(
        f"All {len(errs)} reviewed items the model got wrong against the dataset "
        "label, from both strata, classified by what the reviewers said.\n"
    )

    def classify(r):
        if r["tied"] or r["consensus"] is None:
            return "Ambiguous - reviewers disagreed"
        if r["consensus"] == r["model_pred"]:
            return "Dataset label wrong - humans sided with the model"
        if r["model_pred"] in str(r["accepted_set"]).split("|"):
            return "Defensible - model named a secondary topic"
        return "Genuine model error - humans sided with the dataset"

    labelled = errs.assign(category=errs.apply(classify, axis=1))
    counts = labelled["category"].value_counts()
    out.append("| Category | Count | Share |")
    out.append("|---|---:|---:|")
    for cat, c in counts.items():
        out.append(f"| {cat} | {c} | {fmt_pct(c / len(labelled))} |")
    out.append("")

    real = counts.get("Genuine model error - humans sided with the dataset", 0)
    out.append(
        f"Only {real} of {len(labelled)} reviewed errors ({fmt_pct(real / max(len(labelled), 1))}) "
        "are failures a better model could plausibly fix. The rest are label or "
        "task-definition problems, which is exactly what the reported accuracy "
        "figure hides.\n"
    )
    labelled[
        ["item_id", "stratum", "dataset_label", "model_pred", "consensus", "category"]
    ].to_csv(out_dir / "error_taxonomy.csv", index=False)
    return {k: int(v) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", default="sample")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    sample_dir = HERE / args.sample_dir
    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    key, sheets = load_sheets(sample_dir)
    scored = build_consensus(key, sheets)
    scored.to_csv(out_dir / "scored_items.csv", index=False)

    out = ["# Human Evaluation Report: MathNet Topic Classifier", ""]
    out.append(
        f"{len(key)} validation problems, independently labeled by "
        f"{len(sheets)} reviewers blind to both the dataset label and the "
        "model's prediction.\n"
    )
    summary = {"n_items": int(len(key)), "n_reviewers": int(len(sheets))}
    summary["agreement"] = agreement_section(key, sheets, out)
    summary["accuracy"] = accuracy_section(scored, out)
    summary["taxonomy"] = taxonomy_section(scored, out, out_dir)

    (out_dir / "human_eval_report.md").write_text("\n".join(out), encoding="utf-8")
    (out_dir / "human_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'human_eval_report.md'}")
    print(f"Wrote {out_dir / 'error_taxonomy.csv'} and human_eval_summary.json")


if __name__ == "__main__":
    main()
