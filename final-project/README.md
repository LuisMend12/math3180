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
| `requirements.txt` | Python dependencies |
| `saved_model/mathnet_cnn.pt` | Saved PyTorch model weights (generated after running the notebook) |
| `saved_model/tokenizer.pkl` | Fitted tokenizer |
| `saved_model/label_encoder.pkl` | Label encoder for topic names |

## Getting the Data

The notebook reads MathNet parquet files from a local `MathNet/` folder. Clone it from HuggingFace using `git lfs` (required for large files):

```bash
# Install git-lfs if you don't have it
git lfs install

# Clone the dataset into the final-project folder
git clone https://huggingface.co/datasets/ShadenA/MathNet
```

After cloning, your directory should look like:

```
final-project/
├── MathNet/
│   └── data/
│       └── all/
│           ├── train-00000-of-XXXXX.parquet
│           └── ...
├── mathnet_cnn.ipynb
└── README.md
```

The notebook filters to the **top 10 countries** by row count to keep training fast. A bar chart of the selected countries is shown in cell 3 of the notebook.

## Requirements

**Python 3.11 is required.** TensorFlow does not support Python 3.12+ on Windows, so this notebook uses PyTorch instead. Python 3.11 is the recommended version for full compatibility with all dependencies.

Install dependencies:

```bash
pip install -r requirements.txt
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
