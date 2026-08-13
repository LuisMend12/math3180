# Human Evaluation of the MathNet Topic Classifier

The reported 85.75% validation accuracy measures agreement between the model
and the dataset's label, not whether the model is right. Those are different
claims whenever the label is itself uncertain, and for this task it often is.
This directory contains the protocol and tooling to measure the difference.

## Why this is needed here, specifically

The notebook derives each problem's class from `topics_flat` with
`topics[0].split(">")[0]` — it keeps the **first** topic and discards the rest.
Running `build_review_sample.py` measures what that costs on the 5,447-item
validation set:

| Quantity | Value |
|---|---:|
| Items whose topic list touches more than one of the four classes | 1,440 (26.4%) |
| Model errors against the dataset label | 777 (14.3%) |
| ...where the model's prediction *is* one of that item's own listed topics | 371 (47.7% of errors) |
| Accuracy if any listed topic is accepted | 92.55% |

Nearly half of the model's apparent mistakes name a topic the dataset itself
assigns to the problem, just not in first position. A problem tagged
`[Algebra, Number Theory]` is scored as Algebra, so a Number Theory prediction
is counted wrong even though the dataset agrees it is a Number Theory problem.

This is an argument for review, not a substitute for it: it shows the label is
frequently arbitrary, but only a human can say which topic a given problem
actually belongs to. Hence the protocol below.

## Protocol

**1. Draw a stratified sample.**

```bash
python build_review_sample.py --n-random 150 --n-error 60 --reviewers 3
```

Two strata are drawn and shuffled together into a single sheet:

- `random` (150 items) — a uniform sample of the validation set. All accuracy
  estimates come from this stratum alone, so they stay unbiased.
- `error` (60 items) — extra model errors, used only to characterize *how* the
  model fails. Including these in the accuracy estimate would inflate the error
  rate, which is why the scorer separates them.

Reviewers cannot tell the strata apart; the assignment lives only in
`sample/answer_key.csv`, which they must not see. Note that the key is
committed here for reproducibility, so send reviewers only their own
`review_sheet_reviewerN.csv` and `review_packet.md` rather than pointing them
at this directory.

**2. Review blind.** Each reviewer gets `sample/review_sheet_reviewerN.csv`
and `sample/review_packet.md` (the same problems, formatted for reading). For
each item they record:

| Column | Meaning |
|---|---|
| `primary_topic` | Algebra, Combinatorics, Geometry, or Number Theory |
| `secondary_topic` | A second class if the problem genuinely belongs to two; blank otherwise |
| `confidence` | high / medium / low |
| `notes` | Optional — why the item was hard to place |

Blinding matters. A reviewer shown the model's prediction will anchor to it,
and the resulting agreement number measures suggestibility rather than
correctness. Reviewers also work independently and do not discuss items until
every sheet is submitted, because the point of a second reviewer is an
independent draw.

**3. Score.**

```bash
python score_review.py
```

Writes `results/human_eval_report.md`, `results/error_taxonomy.csv`,
`results/scored_items.csv`, and `results/human_eval_summary.json`.

## What the scorer reports, and why each part is there

**Inter-annotator agreement** (pairwise Cohen's kappa, overall Fleiss' kappa).
This is the human ceiling. If three mathematicians only agree with each other
82% of the time about whether a problem is Algebra or Number Theory, then a
model at 85% against a single arbitrary label is not 15% away from perfect —
it is already near the limit of what the task definition supports. Without this
number there is nothing to compare the model's accuracy *to*.

**Accuracy against the human consensus**, with 95% Wilson intervals, next to
accuracy against the dataset label and the consensus-vs-dataset agreement rate.
The last of these is label quality, and it bounds the first. Wilson intervals
rather than a point estimate because 150 items is a small sample: the interval
is roughly ±6 points, and a report that says "85.8%" from 150 reviewed items
without an interval is overclaiming.

**An error taxonomy** splitting apparent errors four ways: genuine model error,
dataset label wrong, defensible secondary topic, and reviewer disagreement.
This is the analysis that turns "14% error rate" into a statement about what
would actually improve if the model improved.

## Sample size

150 random items gives a 95% interval of about ±6 percentage points near 85%
accuracy. Getting to ±3 would take roughly 500 items per reviewer, which is
usually more review effort than a course project can spend. Two reviewers are
the minimum for any agreement statistic at all; three allow a majority
consensus without tie-breaking, which is why `--reviewers 3` is the default.

Reviewing all 5,447 validation items is not feasible and is not what "verify
each one" has to mean — a stratified sample with stated confidence intervals is
the standard answer, and it is defensible precisely because the interval is
reported rather than hidden.

## Honest limitations to state in the write-up

- Reviewers drawn from classmates are not subject-matter experts; their
  consensus is a proxy for ground truth, not ground truth.
- The four-class scheme forces a single answer onto problems that genuinely
  span topics. Some disagreement is the task's fault, not the reviewers'.
- The error stratum oversamples failures by construction, so no rate may be
  quoted from it. Only the random stratum supports rate claims.
