# MathNet CNN — Math Problem Detection & Understanding

A 1-D CNN classifier trained on the [ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet) dataset to automatically detect and categorize math problems by topic.

## What it does

- Loads the MathNet dataset from HuggingFace
- Trains a text-based CNN to classify math problems across 10+ topic areas
- Outputs the predicted topic and confidence score for any new problem you enter

## Supported Topics

| Topic | Example Problems |
|---|---|
| Calculus | Derivatives, integrals, limits, series, ODEs |
| Algebra | Quadratics, logarithms, systems of equations |
| Geometry | Area, volume, surface area, similar figures |
| Trigonometry | Identities, law of cosines, inverse trig |
| Statistics & Probability | Distributions, hypothesis tests, regression |
| Linear Algebra | Matrix ops, determinants, eigenvalues |
| Number Theory | GCD, modular arithmetic, primes |
| Differential Equations | First/second-order ODEs, IVPs |
| Complex Numbers | Arithmetic, modulus, De Moivre's theorem |
| Combinatorics | Permutations, combinations, pigeonhole |
| Set Theory / Logic | Venn diagrams, proof by induction |

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
print(result["top_3"])             # top 3 topic predictions with scores
```

## Model Architecture

```
Input (token sequence, max_len=128)
  └─ Embedding(20000, 128) → SpatialDropout1D(0.2)
       ├─ Conv1D(128, k=2, relu) → BN → Conv1D(64, k=2, relu) → BN → GlobalMaxPool
       ├─ Conv1D(128, k=3, relu) → BN → Conv1D(64, k=3, relu) → BN → GlobalMaxPool
       ├─ Conv1D(128, k=4, relu) → BN → Conv1D(64, k=4, relu) → BN → GlobalMaxPool
       ├─ Conv1D(128, k=5, relu) → BN → Conv1D(64, k=5, relu) → BN → GlobalMaxPool
       ├─ Conv1D(128, k=6, relu) → BN → Conv1D(64, k=6, relu) → BN → GlobalMaxPool
       ├─ Conv1D(128, k=7, relu) → BN → Conv1D(64, k=7, relu) → BN → GlobalMaxPool
       └─ GlobalAveragePool (soft context branch)
            ↓ Concatenate (6×64 + 128 = 512-dim)
  Dense(512, relu) → BN → Dropout(0.5)
  Dense(256, relu) → BN → Dropout(0.4)
  Dense(128, relu) → BN → Dropout(0.3)
  Dense(64,  relu) → Dropout(0.2)
  Dense(num_classes, softmax)
```

Six parallel Conv1D branches with kernel sizes 2–7 capture n-gram patterns of varying lengths. A global average pooling branch on the raw embedding provides softer contextual signal. The branches merge into a deep classification head with batch normalization and progressive dropout.

**Training:** Adam (lr=1e-3), EarlyStopping (patience=3), ReduceLROnPlateau (factor=0.5, patience=2), up to 20 epochs, batch size 64.

## Course

MATH 3180 — Final Project
