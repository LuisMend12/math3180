# MathNet Math Topic Classification

This final project classifies olympiad-style math problem statements by topic
using the [ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet)
dataset. The main notebook, `mathnet_cnn.ipynb`, loads MathNet parquet shards,
maps detailed hierarchical topic labels into four broad classes, and compares
classical text baselines with neural and pretrained-embedding approaches.

The four prediction classes are:

- Algebra
- Combinatorics
- Geometry
- Number Theory

The current notebook compares six models:

- TF-IDF + Ridge Classifier
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- 1D CNN trained from scratch
- Lightweight Transformer encoder trained from scratch
- Sentence-BERT embeddings + Logistic Regression

In the saved notebook output, the strongest validation result is the
**TF-IDF + Ridge Classifier** at **85.75% validation accuracy**. The Linear SVM
and Logistic Regression baselines are close behind, showing that simple lexical
features are very competitive for this topic classification task.

## Project Files

| File | Description |
|---|---|
| `mathnet_cnn.ipynb` | Main notebook for loading data, preprocessing labels, training models, evaluating results, and saving CNN inference artifacts |
| `requirements.txt` | Core Python dependencies, including `sentence-transformers` for SBERT embeddings |
| `mathnet_dataset.json` | Local dataset reference/export |
| `MathNet/` | Local clone of the MathNet dataset from Hugging Face |
| `saved_model/` | Generated after running the notebook; contains the saved CNN, tokenizer, and label encoder |
| `human_eval/` | Blind human-review protocol and scoring tooling for validating the reported accuracy |

## Human Evaluation

Validation accuracy measures agreement with the dataset's label, which is not
the same as being correct. The label for each problem is the first entry of
`topics_flat`, so problems spanning several topics get an arbitrary primary
class. Measured on the validation set: 26.4% of items touch more than one of
the four classes, and 47.7% of the model's errors predict a topic that is in
that item's own topic list — accuracy rises to 92.55% if any listed topic is
accepted.

`human_eval/` contains a blind review protocol that separates genuine model
errors from label artifacts: a stratified sample, sheets for independent
reviewers who see neither the label nor the prediction, and a scorer reporting
inter-annotator agreement, accuracy against human consensus with Wilson
confidence intervals, and an error taxonomy. See `human_eval/README.md`.

## Dataset

The notebook reads parquet shards from:

```text
MathNet/data/all/train-*.parquet
```

Dataset summary from the notebook run:

| Item | Value |
|---|---:|
| Parquet files loaded | 56 |
| Total rows loaded | 27,817 |
| Rows used after label filtering | 27,231 |
| Training samples | 21,784 |
| Validation samples | 5,447 |
| Input text column | `problem_markdown` |
| Label column | `topics_flat` |
| CNN/Transformer sequence length | 256 tokens |
| CNN/Transformer vocabulary size | 20,000 words |

The original `topics_flat` labels are hierarchical. The notebook maps them into
four broad classes and drops rare or unmapped categories.

| Class | Samples |
|---|---:|
| Geometry | 8,313 |
| Algebra | 7,859 |
| Combinatorics | 5,864 |
| Number Theory | 5,195 |

## Models

### TF-IDF Linear Baselines

The notebook uses TF-IDF unigram and bigram features with three linear
classifiers: Ridge Classifier, Logistic Regression, and Linear SVM. These are
strong baselines because math topic labels often have distinctive vocabulary.
For example, words such as `modulo`, `divisor`, and `congruent` are strong
signals for Number Theory, while `triangle`, `tangent`, and `circumscribed` are
strong signals for Geometry.

These models are fast to train, easy to interpret, and perform best overall in
the current notebook run.

### 1D CNN

The CNN tokenizes problem text, pads each sequence to 256 tokens, then uses an
embedding layer with parallel 1D convolution branches. The convolution filters
learn local phrase patterns that are useful for topic prediction, such as
short mathematical expressions or topic-specific word groups.

The notebook currently saves this CNN model for later inference:

```text
saved_model/mathnet_cnn.keras
saved_model/tokenizer.pkl
saved_model/label_encoder.pkl
```

### Lightweight Transformer

The Transformer model uses the same integer-token sequences as the CNN, but adds
learned positional embeddings and self-attention blocks. This lets the model use
longer-range relationships between words, although in this run it still performs
slightly below the best TF-IDF baselines.

### Sentence-BERT Embeddings

The pretrained embedding model uses `sentence-transformers` with
`all-MiniLM-L6-v2`. Each problem statement is converted into a dense
384-dimensional vector, and a Logistic Regression classifier is trained on top.
This approach tests whether transfer learning from a pretrained language model
helps more than training embeddings from scratch on MathNet alone.

## Model Comparison

Saved validation results from the notebook:

| Model | Validation Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Ridge | 85.75% | 84.77% | 84.74% | 84.74% |
| TF-IDF + Linear SVM | 85.46% | 84.45% | 84.43% | 84.44% |
| TF-IDF + Logistic Regression | 85.22% | 84.30% | 84.12% | 84.18% |
| Transformer from scratch | 84.62% | 83.32% | 83.93% | 83.44% |
| SBERT + Logistic Regression | 84.49% | 83.41% | 83.37% | 83.37% |
| 1D CNN from scratch | 83.70% | 82.39% | 82.90% | 82.41% |

## Model Parameter Counts

| Model | Parameters |
|---|---:|
| TF-IDF + Ridge | 80,004 |
| TF-IDF + Linear SVM | 80,004 |
| TF-IDF + Logistic Regression | 80,004 |
| Transformer from scratch | 2,841,988 |
| SBERT + Logistic Regression | 1,540 classifier params, plus ~22M frozen SBERT params |
| 1D CNN from scratch | 3,395,588 total, 3,392,516 trainable |

The TF-IDF linear baselines use 20,000 TF-IDF features and 4 classes, giving
`20,000 * 4 + 4 = 80,004` learned classifier parameters. The SBERT Logistic
Regression head uses 384-dimensional sentence embeddings and 4 classes, giving
`384 * 4 + 4 = 1,540` learned classifier parameters. The pretrained
`all-MiniLM-L6-v2` encoder is frozen during this experiment and contributes
about 22M pretrained parameters. The CNN and Transformer counts are taken from
the Keras model summaries saved in the notebook output.

## Example CNN Predictions

The notebook tests the saved CNN inference helper on hand-written math
problems. Correct examples include quadratic equations, triangle area,
tangent lines, greatest common divisor, Chinese remainder theorem,
combinations, and pigeonhole principle problems.

Some errors occur when a problem's wording overlaps multiple topics. For
example, algebraic equations or inequalities can be confused with Number Theory
when they contain integer constraints, divisibility language, or number-focused
phrasing.

## How To Run

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
pip install tensorflow keras tf-keras
```

The notebook also includes install cells for Colab-style usage, including
`sentence-transformers` for the SBERT section.

If the MathNet dataset is not already present, clone it into the
`final-project` directory:

```bash
git lfs install
git clone https://huggingface.co/datasets/ShadenA/MathNet
```

Open the notebook:

```bash
jupyter lab mathnet_cnn.ipynb
```

Run the cells from top to bottom. The notebook will load the dataset, map topic
labels, train the comparison models, evaluate validation performance, test
example predictions, and save the CNN artifacts.

## Saved CNN Inference

After running the notebook, the final CNN save cell writes:

```text
saved_model/mathnet_cnn.keras
saved_model/tokenizer.pkl
saved_model/label_encoder.pkl
```

The notebook includes this helper for predicting a new problem topic with the
saved CNN pipeline:

```python
result = predict_topic("Find the greatest common divisor of 252 and 198.")
print(result["predicted_topic"])
print(result["confidence"])
print(result["top_3"])
```

Example top-3 output:

```text
[Number Theory] Find the greatest common divisor of 252 and 198 using the Euclidean algorithm
  Number Theory          98.97%
  Algebra                 1.00%
  Combinatorics           0.03%
```

## Conclusion

The project shows that math problem text contains enough signal to classify
problems into broad topic areas with useful accuracy. The strongest models in
this run are the TF-IDF linear baselines, especially Ridge and Linear SVM. This
is an important result: for structured topic classification with clear
vocabulary cues, classical text models can outperform more complex neural
models trained from scratch.

The CNN and Transformer remain useful comparisons because they show how
sequence-based neural models handle the same data. The SBERT experiment adds a
transfer-learning baseline using pretrained sentence embeddings, but in this
run it does not outperform the strongest TF-IDF models.

Future improvements could save the best-performing TF-IDF pipeline for
inference, tune the pretrained embedding model, try math- or science-specific
embedding models, include solution text or diagram features, and improve
handling for rare or overlapping topic categories.
