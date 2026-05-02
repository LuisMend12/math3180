# MathNet Math Topic Classification

This final project classifies olympiad-style math problems by topic using the
[ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet) dataset.
The main notebook, `mathnet_cnn.ipynb`, loads problem text from MathNet,
maps detailed topic labels into four broad classes, and compares three text
classification models:

- TF-IDF with a Ridge Classifier baseline
- A 1D convolutional neural network
- A lightweight Transformer encoder

The strongest validation result in the notebook is the TF-IDF Ridge baseline at
**85.75% validation accuracy**, followed closely by the Transformer at
**85.40%**. The final inference helper and saved artifacts use the 1D CNN model.

## Project Files

| File | Description |
|---|---|
| `mathnet_cnn.ipynb` | Main notebook for loading data, preprocessing labels, training models, evaluating results, and saving inference artifacts |
| `requirements.txt` | Core Python dependencies |
| `mathnet_dataset.json` | Local dataset export/reference file |
| `MathNet/` | Local clone of the MathNet dataset from Hugging Face |
| `saved_model/` | Generated after running the notebook; contains the saved CNN, tokenizer, and label encoder |

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

### Ridge Classifier Baseline

The baseline uses TF-IDF features with a multi-class Ridge Classifier. This is a
strong comparison point because it is fast, interpretable, and well suited for
text classification.

Validation results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Algebra | 0.84 | 0.85 | 0.84 | 1,572 |
| Combinatorics | 0.80 | 0.83 | 0.81 | 1,173 |
| Geometry | 0.96 | 0.94 | 0.95 | 1,663 |
| Number Theory | 0.80 | 0.78 | 0.79 | 1,039 |
| Accuracy |  |  | 0.86 | 5,447 |
| Macro avg | 0.85 | 0.85 | 0.85 | 5,447 |
| Weighted avg | 0.86 | 0.86 | 0.86 | 5,447 |

### 1D CNN

The CNN tokenizes problem text, pads each sequence to 256 tokens, then uses an
embedding layer with parallel 1D convolution branches. The final notebook cell
saves this model for later inference.

Validation results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Algebra | 0.91 | 0.72 | 0.80 | 1,572 |
| Combinatorics | 0.76 | 0.80 | 0.78 | 1,173 |
| Geometry | 0.95 | 0.91 | 0.93 | 1,663 |
| Number Theory | 0.65 | 0.87 | 0.74 | 1,039 |
| Accuracy |  |  | 0.82 | 5,447 |
| Macro avg | 0.82 | 0.82 | 0.81 | 5,447 |
| Weighted avg | 0.84 | 0.82 | 0.83 | 5,447 |

### Transformer Encoder

The Transformer model uses the same integer-token sequences as the CNN, but
adds positional information and self-attention blocks. It performed slightly
below the Ridge baseline and above the CNN in best validation accuracy.

## Model Comparison

| Model | Parameters | Best Validation Accuracy |
|---|---:|---:|
| Ridge Classifier (TF-IDF) | 80,000 | 85.75% |
| 1D CNN (text) | 3,668,292 | 83.17% |
| Transformer (text) | 2,841,988 | 85.40% |

## Example CNN Predictions

The notebook tests the saved CNN inference helper on 30 hand-written math
problems. It correctly classifies 26 of the 30 examples.

Correct examples include:

| Example Type | Predicted | Confidence |
|---|---|---:|
| Quadratic equation | Algebra | 94.2% |
| Polynomial factoring | Algebra | 94.0% |
| Triangle area | Geometry | 100.0% |
| Tangent lines | Geometry | 100.0% |
| Greatest common divisor | Number Theory | 99.0% |
| Chinese remainder theorem | Number Theory | 95.0% |
| Combinations | Combinatorics | 99.6% |
| Pigeonhole principle | Combinatorics | 98.1% |

Misclassified examples include:

| Expected | Predicted | Confidence | Example Type |
|---|---|---:|---|
| Algebra | Number Theory | 75.0% | System of equations |
| Algebra | Number Theory | 55.7% | Exponential equation |
| Algebra | Number Theory | 98.0% | Absolute value inequality |
| Combinatorics | Number Theory | 99.4% | Inclusion-exclusion |

## How To Run

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
pip install tensorflow keras tf-keras
```

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

Run the cells from top to bottom. The notebook will load the dataset, train the
models, evaluate the validation split, test example predictions, and save the
CNN artifacts.

## Saved CNN Inference

After running the notebook, the final cell writes:

```text
saved_model/mathnet_cnn.keras
saved_model/tokenizer.pkl
saved_model/label_encoder.pkl
```

The notebook includes this helper for predicting a new problem topic:

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
problems into broad topic areas with useful accuracy. The TF-IDF Ridge baseline
was the best-performing model in this run, which is a useful reminder that
classical text models can be very competitive on structured classification
tasks. The Transformer was close behind, while the 1D CNN was saved as the
notebook's reusable inference model.

Future improvements could save the best-performing model for inference, tune the
Transformer more carefully, include solution text or diagram features, and add
better handling for rare topic categories.
