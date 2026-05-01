# MathNet CNN — Math Problem Detection & Understanding

A multi-model classifier trained on the [ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet) dataset to automatically detect and categorize math problems by topic. The notebook trains and compares three models: a text-based 1-D CNN, a pretrained EfficientNetB0, and a pretrained MobileNetV2-0.35 — all using TensorFlow/Keras.

## What it does

- Loads the MathNet dataset (parquet files) and filters to the top N countries by row count (default: 1 — United States, ~4,819 rows out of 27,817 total)
- Trains a **1-D CNN on problem text** (`problem_markdown`) to classify math topics
- Trains **EfficientNetB0** and **MobileNetV2-0.35** (both frozen ImageNet bases) on the problem figures
- Compares all three models side-by-side on parameter count and validation accuracy
- Outputs the predicted topic and confidence score for any new problem you enter

## Topics

The model classifies problems using the full hierarchical `topics_flat` labels from the MathNet dataset. With the default single-country filter there are **2,509 unique fine-grained topic strings**, such as:

- `Algebra > Equations and Inequalities > QM-AM-GM-HM Inequalities`
- `Geometry > Solid Geometry > Volume`
- `Statistics > Probability > Counting Methods > Combinations`
- `Number Theory > Divisibility / Factorization > Prime numbers`

To increase variety and reduce the number of classes, increase `nlargest(N)` in cell 2 to include more countries.

## Files

| File | Description |
|---|---|
| `mathnet_cnn.ipynb` | Main notebook — data loading, training, evaluation, and inference |
| `requirements.txt` | Python dependencies |
| `saved_model/mathnet_cnn.keras` | Saved 1-D CNN model weights (generated after running the notebook) |
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

The notebook filters to the **top N countries** by row count to keep training fast. Change `nlargest(1)` in cell 2 to include more countries and more training data.

## Requirements

**Python 3.11 is required.** TensorFlow 2.x does not support Python 3.12+ on Windows. Python 3.11 is the recommended version for full compatibility with all dependencies.

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

### Model 1 — 1-D CNN on text (primary model, used for `predict_topic`)

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

**Training:** Adam (lr=1e-4), EarlyStopping (patience=3), ReduceLROnPlateau (factor=0.5, patience=2), up to 60 epochs, batch size 64.

### Model 2 — EfficientNetB0 on problem figures (large pretrained)

ImageNet-pretrained EfficientNetB0 with its convolutional base frozen. A small classification head is trained on the problem figures extracted from the dataset.

```
Input (224×224×3 image)
  └─ EfficientNetB0 base (frozen, ImageNet weights)
       └─ GlobalAveragePooling2D → BN → Dense(256, relu) → Dropout(0.5)
            └─ Dense(num_classes, softmax)
```

**Training:** Adam (lr=1e-4), EarlyStopping (patience=4), ReduceLROnPlateau (factor=0.5, patience=2), up to 20 epochs, batch size 32.

### Model 3 — MobileNetV2-0.35 on problem figures (small pretrained)

A compact MobileNetV2 (`alpha=0.35`, ~400K parameters) pretrained on ImageNet. The base is frozen; only the classification head trains.

```
Input (224×224×3 image)
  └─ MobileNetV2-0.35 base (frozen, ImageNet weights)
       └─ GlobalAveragePooling2D → BN → Dense(128, relu) → Dropout(0.4)
            └─ Dense(num_classes, softmax)
```

**Training:** Adam (lr=1e-4), EarlyStopping (patience=4), ReduceLROnPlateau (factor=0.5, patience=2), up to 20 epochs, batch size 32.

### Memory-efficient image pipeline

All image preprocessing is done with a `tf.data.Dataset` pipeline that applies `preprocess_input` **per batch** instead of upfront, avoiding the allocation of a full ~6 GB preprocessed array in RAM. The raw decoded images (`X_img_train`, shape `[11079, 224, 224, 3]`) stay in memory once; preprocessing runs on the fly during training.

## Course

MATH 3180 — Final Project
