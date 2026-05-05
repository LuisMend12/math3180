from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (
    CATEGORY_MAP,
    CLS_TOKEN,
    EXPECTED_LABEL_COUNTS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MASK_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    SPECIAL_TOKENS,
    TinyConfig,
    UNK_TOKEN,
)


def set_parallel_env() -> None:
    cpu_count = os.cpu_count() or 4
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, cpu_count - 1)))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cpu_count - 1)))


def extract_top_label(topics_raw) -> str | None:
    try:
        topics = topics_raw if isinstance(topics_raw, list) else ast.literal_eval(str(topics_raw))
        if topics:
            top = topics[0].split(">")[0].strip()
            return CATEGORY_MAP.get(top)
    except Exception:
        return None
    return None


def load_mathnet_rows(data_path: Path) -> tuple[list[str], np.ndarray, dict[str, int]]:
    rows = json.loads(data_path.read_text())
    texts: list[str] = []
    labels: list[int] = []
    counts = {k: 0 for k in EXPECTED_LABEL_COUNTS}
    for row in rows:
        label = extract_top_label(row.get("topics_flat"))
        if label is None:
            continue
        counts[label] += 1
        texts.append(str(row.get("problem_markdown", "")))
        labels.append(LABEL_TO_ID[label])

    if counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected label counts: {counts}; expected {EXPECTED_LABEL_COUNTS}")
    return texts, np.asarray(labels, dtype=np.int64), counts


def stratified_split(
    texts: list[str], labels: np.ndarray, seed: int
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels), dtype=np.int64)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    def pack(idx: np.ndarray, y: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        return [texts[int(i)] for i in idx], y.astype(np.int64), idx.astype(np.int64)

    return {
        "train": pack(train_idx, y_train),
        "val": pack(val_idx, y_val),
        "test": pack(test_idx, y_test),
    }


def normalize_for_grouping(text: str) -> str:
    text = text.lower()
    text = re.sub(r"!\[\]\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def grouped_stratified_split(
    texts: list[str], labels: np.ndarray, seed: int
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    indices = np.arange(len(labels), dtype=np.int64)
    groups = np.asarray(
        [hashlib.sha1(normalize_for_grouping(text).encode()).hexdigest() for text in texts],
        dtype=object,
    )

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(outer.split(indices, labels, groups))
    temp_labels = labels[temp_idx]
    temp_groups = groups[temp_idx]

    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
    val_rel, test_rel = next(inner.split(temp_idx, temp_labels, temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    def pack(idx: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        idx = idx.astype(np.int64)
        return [texts[int(i)] for i in idx], labels[idx].astype(np.int64), idx

    return {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }


def build_tokenizer(vocab_size: int, max_len: int):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer

    try:
        model = BPE(unk_token=UNK_TOKEN, byte_fallback=True)
    except TypeError:
        model = BPE(unk_token=UNK_TOKEN)

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.enable_truncation(max_length=max_len)
    return tokenizer, trainer


def train_tokenizer(train_texts: Iterable[str], vocab_size: int, max_len: int):
    from tokenizers.processors import TemplateProcessing

    tokenizer, trainer = build_tokenizer(vocab_size=vocab_size, max_len=max_len)
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
        special_tokens=[
            (CLS_TOKEN, tokenizer.token_to_id(CLS_TOKEN)),
            (SEP_TOKEN, tokenizer.token_to_id(SEP_TOKEN)),
        ],
    )
    tokenizer.enable_truncation(max_length=max_len)
    return tokenizer


def _pad_encoding(ids: list[int], pad_id: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(ids) > max_len:
        ids = ids[:max_len]
    arr = np.full(max_len, pad_id, dtype=np.uint16)
    mask = np.zeros(max_len, dtype=np.uint8)
    n = len(ids)
    arr[:n] = np.asarray(ids, dtype=np.uint16)
    mask[:n] = 1
    return arr, mask


def encode_texts(tokenizer, texts: list[str], max_len: int, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    chunk_size = max(128, len(texts) // workers)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    def encode_chunk(chunk: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = tokenizer.encode_batch(chunk)
        ids_out = np.empty((len(encodings), max_len), dtype=np.uint16)
        mask_out = np.empty((len(encodings), max_len), dtype=np.uint8)
        for i, enc in enumerate(encodings):
            ids_out[i], mask_out[i] = _pad_encoding(enc.ids, pad_id, max_len)
        return ids_out, mask_out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(encode_chunk, chunks))
    ids = np.concatenate([p[0] for p in parts], axis=0)
    masks = np.concatenate([p[1] for p in parts], axis=0)
    return ids, masks


def build_math_token_mask(tokenizer) -> np.ndarray:
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    mask = np.zeros(size, dtype=np.uint8)
    common_vars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for token, idx in vocab.items():
        if token in SPECIAL_TOKENS:
            continue
        try:
            decoded = tokenizer.decode([idx]).strip()
        except Exception:
            decoded = token.strip()
        stripped = decoded.replace(" ", "")
        if not stripped:
            continue
        if any(ch.isdigit() for ch in stripped) or (len(stripped) == 1 and stripped in common_vars):
            mask[idx] = 1
    return mask


def save_split_npz(path: Path, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
    np.savez_compressed(
        path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels.astype(np.int64),
        indices=indices.astype(np.int64),
    )


def prepare_dataset(
    data_path: Path,
    run_dir: Path,
    vocab_size: int = 4096,
    max_len: int = 256,
    seed: int = 42,
    group_duplicates: bool = False,
) -> dict:
    set_parallel_env()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)

    texts, labels, counts = load_mathnet_rows(data_path)
    split_kind = "grouped_stratified" if group_duplicates else "stratified"
    splits = grouped_stratified_split(texts, labels, seed=seed) if group_duplicates else stratified_split(texts, labels, seed=seed)
    train_texts, _, _ = splits["train"]
    tokenizer = train_tokenizer(train_texts, vocab_size=vocab_size, max_len=max_len)
    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    special_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    cfg = TinyConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=max_len,
        pad_id=special_ids[PAD_TOKEN],
        cls_id=special_ids[CLS_TOKEN],
        sep_id=special_ids[SEP_TOKEN],
        mask_id=special_ids[MASK_TOKEN],
    )

    split_summary = {}
    for split_name, (split_texts, split_labels, split_indices) in splits.items():
        input_ids, attention_mask = encode_texts(tokenizer, split_texts, max_len=max_len)
        save_split_npz(
            run_dir / "data" / f"{split_name}.npz",
            input_ids,
            attention_mask,
            split_labels,
            split_indices,
        )
        split_summary[split_name] = {
            "size": int(len(split_labels)),
            "counts": {ID_TO_LABEL[i]: int((split_labels == i).sum()) for i in sorted(ID_TO_LABEL)},
        }

    math_token_mask = build_math_token_mask(tokenizer)
    np.save(run_dir / "data" / "math_token_mask.npy", math_token_mask)

    meta = {
        "data_path": str(data_path),
        "seed": seed,
        "split_kind": split_kind,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": cfg.vocab_size,
        "max_len": max_len,
        "label_counts": counts,
        "splits": split_summary,
        "special_ids": special_ids,
        "model_config": cfg.to_dict(),
    }
    (run_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (run_dir / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    return meta


def load_npz_split(run_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = run_dir / "data" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared split: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def load_model_config(run_dir: Path) -> TinyConfig:
    return TinyConfig.from_dict(json.loads((run_dir / "model_config.json").read_text()))
