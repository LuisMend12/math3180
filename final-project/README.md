# MathNet CNN — Math Problem Detection & Understanding

A 1-D CNN classifier trained on the [ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet) dataset to automatically detect and categorize math problems by topic.

## What it does

- Loads the MathNet dataset from HuggingFace
- Trains a text-based CNN to classify math problems (e.g. algebra, calculus, geometry)
- Outputs the predicted topic and confidence score for any new problem you enter

## Files

| File | Description |
|---|---|
| `mathnet_cnn.ipynb` | Main notebook — data loading, training, evaluation, and inference |
| `saved_model/mathnet_cnn.keras` | Saved model (generated after running the notebook) |
| `saved_model/tokenizer.pkl` | Fitted tokenizer |
| `saved_model/label_encoder.pkl` | Label encoder for topic names |

## Setup

```bash
pip install datasets transformers torch torchvision matplotlib scikit-learn seaborn pandas numpy tensorflow
```

## Usage

Open `mathnet_cnn.ipynb` and run all cells. To predict the topic of a new problem:

```python
result = predict_topic("Find the derivative of f(x) = 3x^2 + 2x - 5")
print(result["predicted_topic"])   # e.g. "Calculus"
print(result["confidence"])        # e.g. 0.94
```

## Model Architecture

```
Embedding → [Conv1D(k=3), Conv1D(k=4), Conv1D(k=5)] → GlobalMaxPool
         → Concatenate → Dropout(0.4) → Dense(128) → Dropout(0.3) → Softmax
```

Three parallel convolution branches capture short, medium, and longer n-gram patterns in the problem text before merging for classification.

## Course

MATH 3180 — Final Project
