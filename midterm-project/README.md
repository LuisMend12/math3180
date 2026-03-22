# 📧 Spam Email Classification Using Bernoulli Naive Bayes

Machine learning project for detecting spam emails using the **Spambase dataset** and the **Bernoulli Naive Bayes classification model**.

---

# 📌 Overview

Spam email detection is a fundamental problem in machine learning and cybersecurity. Email providers must automatically distinguish between legitimate messages and unwanted spam in order to protect users and maintain effective communication systems.

In this project, we investigate whether the **presence or absence of certain words and characters in an email** can be used to accurately classify messages as spam.

To accomplish this, we implement a **Bernoulli Naive Bayes classifier**, which models each feature as a **binary variable indicating whether a word or character appears in the email**.

---

# 🎯 Objectives

The goal of this project is to:

* Explore the **Spambase dataset**
* Convert numerical predictors into **binary indicators**
* Train a **Bernoulli Naive Bayes classifier**
* Evaluate model performance using classification metrics
* Interpret which features are most indicative of spam

---

# 📊 Dataset

We use the **Spambase dataset**, a well-known dataset for spam detection research.

Dataset source:
UCI Machine Learning Repository

### Dataset Characteristics

| Property        | Value                         |
| --------------- | ----------------------------- |
| Total Emails    | 4601                          |
| Spam Emails     | 1813 (39.4%)                  |
| Non-Spam Emails | 2788 (60.6%)                  |
| Features        | 57                            |
| Target Variable | spam (0 = not spam, 1 = spam) |

### Feature Types

The predictors represent statistics extracted from each email.

#### Word Frequency Features (48)

Examples:

* `word_freq_make`
* `word_freq_address`
* `word_freq_free`
* `word_freq_money`

Each represents:

```
100 × (number of times the word appears / total words in email)
```

Spam emails commonly contain words like:

* free
* money
* credit
* offer
* remove
* guarantee

---

#### Character Frequency Features (6)

Examples:

* `char_freq_!`
* `char_freq_$`
* `char_freq_#`

Spam messages often contain excessive punctuation such as:

```
!!! FREE MONEY !!!
$$$$$
```

---

#### Capital Letter Statistics (3)

These capture patterns of uppercase text.

* `capital_run_length_average`
* `capital_run_length_longest`
* `capital_run_length_total`

Spam emails often use **all caps** to grab attention.

---

# 🧠 Model: Bernoulli Naive Bayes

The **Bernoulli Naive Bayes model** is appropriate when features represent **binary outcomes** such as word presence.

### Binary Feature Transformation

Each feature is converted to:

```
Xj = 1  if feature appears
Xj = 0  otherwise
```

This allows the model to treat predictors as **Bernoulli random variables**.

---

# 📐 Mathematical Background

### Bayes Rule

[
P(Y=k|X) = \frac{P(X|Y=k)P(Y=k)}{P(X)}
]

Where:

* (P(Y=k)) is the **prior probability**
* (P(X|Y=k)) is the **likelihood**
* (P(Y=k|X)) is the **posterior probability**

---

### Naive Independence Assumption

The model assumes predictors are conditionally independent given the class:

[
P(X|Y=k) = \prod_{j=1}^{p} P(X_j|Y=k)
]

---

### Bernoulli Likelihood

[
P(X_j = 1|Y=k) = \theta_{jk}
]

[
P(X|Y=k) = \prod_{j=1}^{p} \theta_{jk}^{x_j}(1-\theta_{jk})^{1-x_j}
]

---

### Classification Rule (Log Form)

[
\hat{y} =
\arg\max_k
\left[
\log P(Y=k) +
\sum x_j\log\theta_{jk} +
(1-x_j)\log(1-\theta_{jk})
\right]
]

Logs are used to **avoid numerical underflow** and simplify computation.

---

# ⚙️ Data Preparation

The preprocessing pipeline includes:

1. Loading the dataset
2. Converting features to binary indicators
3. Splitting data into **training and testing sets**
4. Using **stratified sampling** to preserve class distribution

Example code:

```python
X = df.drop(columns="spam")
y = df["spam"]

X_binary = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y, test_size=0.3, random_state=42, stratify=y
)
```

---

# 📊 Exploratory Data Analysis

EDA helps understand patterns in the dataset.

### Class Distribution

Visualizing the number of spam vs non-spam emails.

### Feature Presence Analysis

Comparing the proportion of emails containing words such as:

* free
* money
* dollar sign ($)

Spam emails tend to contain **promotional or urgent language**.

---

# 🤖 Model Implementation

The Bernoulli Naive Bayes classifier is trained on the binary dataset.

```python
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)
```

The model estimates:

* prior class probabilities
* feature likelihoods for spam and non-spam emails

---

# 📈 Model Evaluation

We evaluate model performance using several metrics.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score

Example code:

```python
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))
print("F1:", f1_score(y_test,y_pred))
```

---

# 🔍 Results Interpretation

The model achieved strong classification performance on the test dataset.

Words frequently associated with promotions and advertisements were strong indicators of spam.

Examples of predictive signals:

* presence of "free"
* presence of "money"
* excessive punctuation
* capitalized phrases

Despite the independence assumption, Naive Bayes often performs well in practice.

---

# ⚠️ Limitations

Several limitations should be considered:

* The **independence assumption** is unrealistic in real text data.
* Binary conversion removes **frequency information**.
* Some features may be highly **correlated**.
* The dataset reflects **email patterns from HP employees**, which may limit generalization.

---

# 🚀 Future Work

Possible improvements include:

* Comparing with **Multinomial Naive Bayes**
* Applying **Logistic Regression**
* Performing **feature selection**
* Using **cross-validation**
* Training models on **raw text data**

---

# 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 🤖 AI Tools Disclosure

ChatGPT was used to help:

* organize the project structure
* draft explanations of Bernoulli Naive Bayes
* suggest visualization and modeling strategies

All analysis decisions and interpretations were reviewed and finalized by the project group.

---

# 👥 Contributors

Example structure:

| Name      | Contribution                                     |
| --------- | ------------------------------------------------ |
| Luis Mendez | Data preprocessing and exploratory analysis      |
| Student 2 | Mathematical background and model implementation |
| Student 3 | Results interpretation and presentation          |
| Luke Rigoglioso |  |

---

# 📚 References

* UCI Machine Learning Repository – Spambase Dataset
* Scikit-learn Documentation
* Machine Learning textbooks on Naive Bayes classification


=======
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
>>>>>>> origin/newmodifications
