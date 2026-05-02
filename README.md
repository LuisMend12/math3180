# MATH 3180 - Mathematics for Machine Learning

This repository contains two machine learning projects developed for MATH 3180.
Both projects connect mathematical modeling ideas from the course with working
Python implementations, evaluation metrics, and reproducible notebooks.

## Projects

| Folder | Notebook | Topic |
|---|---|---|
| `midterm-project/` | `First_Project_Assignment.ipynb` | Spam email classification with Bernoulli Naive Bayes |
| `final-project/` | `mathnet_cnn.ipynb` | Math problem topic classification with text models |

## Midterm Project: Spam Email Classification

The midterm project uses the UCI Spambase dataset to classify emails as spam or
not spam. The notebook converts numerical email features into binary indicators
and evaluates several Bernoulli Naive Bayes pipelines.

Main ideas covered:

- Bernoulli Naive Bayes classification
- Bayes' rule and log-posterior scoring
- Binary feature transformations
- Train/test splitting with stratification
- Threshold tuning and ablation experiments
- Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices

Latest saved experiment summary:

| Best Model | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| `global_full_f1` | 0.9037 | 0.8735 | 0.9624 |

Important files:

- `midterm-project/First_Project_Assignment.ipynb`: notebook version of the experiment suite
- `midterm-project/main.py`: script version of the experiment suite
- `midterm-project/spambase/spambase.data`: dataset
- `midterm-project/results/`: generated summaries, plots, and feature log-odds

## Final Project: MathNet Topic Classification

The final project uses the `ShadenA/MathNet` olympiad-style math problem dataset
to classify problem statements into four broad topics:

- Algebra
- Combinatorics
- Geometry
- Number Theory

The notebook loads MathNet parquet shards, maps detailed topic labels into the
four topic groups, trains multiple text classifiers, compares validation
performance, tests predictions on hand-written examples, and saves the CNN model
artifacts for later inference.

Models compared:

| Model | Best Validation Accuracy |
|---|---:|
| TF-IDF + Ridge Classifier | 85.75% |
| 1D CNN | 83.17% |
| Lightweight Transformer Encoder | 85.40% |

Important files:

- `final-project/mathnet_cnn.ipynb`: main final project notebook
- `final-project/README.md`: detailed final project writeup
- `final-project/requirements.txt`: final project dependencies
- `final-project/mathnet_dataset.json`: local dataset reference/export
- `final-project/MathNet/`: local MathNet dataset clone

## Setup

Create and activate a Python environment, then install dependencies for the
project you want to run.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For the midterm project:

```bash
pip install -r midterm-project/requirements.txt
```

For the final project:

```bash
pip install -r final-project/requirements.txt
pip install tensorflow keras tf-keras
```

## How To Run

Run the midterm script from the repository root:

```bash
python midterm-project/main.py
```

Or open either notebook in Jupyter:

```bash
jupyter lab
```

Then open:

- `midterm-project/First_Project_Assignment.ipynb`
- `final-project/mathnet_cnn.ipynb`

If the MathNet dataset is not present for the final project, clone it into
`final-project/`:

```bash
cd final-project
git lfs install
git clone https://huggingface.co/datasets/ShadenA/MathNet
```

## Repository Structure

```text
.
├── README.md
├── midterm-project/
│   ├── First_Project_Assignment.ipynb
│   ├── README.md
│   ├── main.py
│   ├── requirements.txt
│   ├── spambase/
│   └── results/
└── final-project/
    ├── mathnet_cnn.ipynb
    ├── README.md
    ├── requirements.txt
    ├── mathnet_dataset.json
    └── MathNet/
```

## Course Themes

These projects apply several mathematical foundations of machine learning:

- Probability and conditional modeling
- Linear classification
- Regularization
- Optimization
- Feature engineering
- Model evaluation and comparison
