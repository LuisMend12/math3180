# MATH 3180 - Mathematics for Machine Learning

This repository contains two machine learning projects for MATH 3180. Each
project includes a notebook, dataset files or dataset instructions, model
training code, evaluation metrics, and a project-specific README.

## Projects

| Project | Folder | Main Notebook | Task | Best Saved Result |
|---|---|---|---|---:|
| Midterm Project | `midterm-project/` | `First_Project_Assignment.ipynb` | Spam email classification | 90.37% test accuracy |
| Final Project | `final-project/` | `mathnet_cnn.ipynb` | Math problem topic classification | 85.75% validation accuracy |

## Project 1: Spam Email Classification

The midterm project uses the UCI Spambase dataset to classify emails as spam or
not spam. The model converts numerical email features into binary indicators and
uses Bernoulli Naive Bayes to estimate whether a message is spam.

Main topics:

- Bernoulli Naive Bayes classification
- Bayes' rule and log-posterior scoring
- Binary feature transformations
- Stratified train/test splitting
- Threshold tuning and ablation experiments
- Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices

Latest saved experiment summary:

| Best Model | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| `global_full_f1` | 0.9037 | 0.8735 | 0.9624 |

Important files:

- `midterm-project/README.md`: detailed project writeup
- `midterm-project/First_Project_Assignment.ipynb`: notebook version
- `midterm-project/main.py`: script version
- `midterm-project/spambase/spambase.data`: dataset
- `midterm-project/results/`: saved plots, summaries, and metrics

## Project 2: MathNet Topic Classification

The final project uses the `ShadenA/MathNet` olympiad-style math problem
dataset to classify problem statements into four broad topics:

- Algebra
- Combinatorics
- Geometry
- Number Theory

The notebook loads MathNet parquet shards, maps detailed topic labels into the
four broad groups, compares several text classification models, evaluates
validation performance, tests hand-written examples, and saves CNN inference
artifacts.

Models compared:

| Model | Validation Accuracy | Parameters |
|---|---:|---:|
| TF-IDF + Ridge Classifier | 85.75% | 80,004 |
| TF-IDF + Linear SVM | 85.46% | 80,004 |
| TF-IDF + Logistic Regression | 85.22% | 80,004 |
| Lightweight Transformer Encoder | 84.62% | 2,841,988 |
| Sentence-BERT + Logistic Regression | 84.49% | 1,540 classifier params, plus ~22M frozen SBERT params |
| 1D CNN | 83.70% | 3,395,588 total, 3,392,516 trainable |

Parameter-count notes:

- The midterm Bernoulli Naive Bayes model has 116 learned parameters: 2 class
  log priors plus 57 feature log probabilities for each of 2 classes.
- The TF-IDF linear baselines use 20,000 TF-IDF features and 4 output classes,
  so each classifier has `20,000 * 4 + 4 = 80,004` learned weights/biases.
- The Sentence-BERT classifier head uses 384-dimensional embeddings and 4
  classes, so the Logistic Regression head has `384 * 4 + 4 = 1,540`
  learned parameters. The pretrained `all-MiniLM-L6-v2` encoder is frozen and
  contributes about 22M pretrained parameters.

Important files:

- `final-project/README.md`: detailed project writeup
- `final-project/mathnet_cnn.ipynb`: main notebook
- `final-project/requirements.txt`: final project dependencies
- `final-project/mathnet_dataset.json`: local dataset reference/export
- `final-project/MathNet/`: local MathNet dataset clone

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies for the project you want to run.

Midterm project:

```bash
pip install -r midterm-project/requirements.txt
```

Final project:

```bash
pip install -r final-project/requirements.txt
pip install tensorflow keras tf-keras
```

If the MathNet dataset is not already present, clone it into `final-project/`:

```bash
cd final-project
git lfs install
git clone https://huggingface.co/datasets/ShadenA/MathNet
```

## How To Run

Run the midterm script from the repository root:

```bash
python midterm-project/main.py
```

Or open the notebooks in Jupyter:

```bash
jupyter lab
```

Then open:

- `midterm-project/First_Project_Assignment.ipynb`
- `final-project/mathnet_cnn.ipynb`

## Repository Structure

```text
.
|-- README.md
|-- midterm-project/
|   |-- README.md
|   |-- First_Project_Assignment.ipynb
|   |-- main.py
|   |-- requirements.txt
|   |-- spambase/
|   `-- results/
`-- final-project/
    |-- README.md
    |-- mathnet_cnn.ipynb
    |-- requirements.txt
    |-- mathnet_dataset.json
    `-- MathNet/
```

## Course Themes

Across the two projects, the repository applies several mathematical foundations
of machine learning:

- Probability and conditional modeling
- Linear classification
- Feature engineering
- Regularization
- Optimization
- Model evaluation and comparison
