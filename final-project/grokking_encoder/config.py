from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

LABEL_TO_ID = {
    "Algebra": 0,
    "Combinatorics": 1,
    "Geometry": 2,
    "Number Theory": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

EXPECTED_LABEL_COUNTS = {
    "Geometry": 8313,
    "Algebra": 7859,
    "Combinatorics": 5864,
    "Number Theory": 5195,
}

CATEGORY_MAP = {
    "Algebra": "Algebra",
    "Number Theory": "Number Theory",
    "Geometry": "Geometry",
    "Trigonometry": "Geometry",
    "Discrete Mathematics": "Combinatorics",
    "Combinatorics": "Combinatorics",
}


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    ff_dim: int = 192
    dropout: float = 0.15
    num_classes: int = 4
    pad_id: int = 0
    cls_id: int = 2
    sep_id: int = 3
    mask_id: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TinyConfig":
        return cls(**data)


def default_run_dir() -> Path:
    return Path("runs") / "tiny_mathnet_grokking"
