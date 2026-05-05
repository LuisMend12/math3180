from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F

from .config import ID_TO_LABEL, TinyConfig
from .data import load_model_config, load_npz_split
from .model import TinyMathBert, assert_under_parameter_budget, count_parameters


def get_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mps_memory() -> dict[str, int]:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        out = {}
        for name in ["current_allocated_memory", "driver_allocated_memory"]:
            fn = getattr(torch.mps, name, None)
            if callable(fn):
                try:
                    out[name] = int(fn())
                except Exception:
                    pass
        return out
    return {}


def process_memory() -> dict[str, int]:
    proc = psutil.Process(os.getpid())
    return {"rss": int(proc.memory_info().rss)}


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def load_device_split(run_dir: Path, split: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = load_npz_split(run_dir, split)
    return {
        "input_ids": torch.as_tensor(data["input_ids"].astype(np.int64), dtype=torch.long, device=device),
        "attention_mask": torch.as_tensor(data["attention_mask"].astype(np.bool_), dtype=torch.bool, device=device),
        "labels": torch.as_tensor(data["labels"], dtype=torch.long, device=device),
    }


def load_cpu_split(run_dir: Path, split: str) -> dict[str, torch.Tensor]:
    data = load_npz_split(run_dir, split)
    return {
        "input_ids": torch.as_tensor(data["input_ids"].astype(np.int64), dtype=torch.long),
        "attention_mask": torch.as_tensor(data["attention_mask"].astype(np.bool_), dtype=torch.bool),
        "labels": torch.as_tensor(data["labels"], dtype=torch.long),
    }


def iter_device_batches(data: dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> dict[str, torch.Tensor]:
    n = int(data["labels"].shape[0])
    if shuffle:
        order = torch.randperm(n, device=data["labels"].device)
    else:
        order = torch.arange(n, device=data["labels"].device)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {key: value[idx] for key, value in data.items()}


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: TinyConfig,
    mlm_prob: float,
    math_mask_prob: float,
    math_token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = attention_mask & (input_ids != cfg.pad_id) & (input_ids != cfg.cls_id) & (input_ids != cfg.sep_id)
    random_values = torch.rand(input_ids.shape, device=input_ids.device)
    mlm_positions = (random_values < mlm_prob) & valid
    if math_token_mask is not None and math_mask_prob > 0:
        math_positions = math_token_mask[input_ids] & valid
        mlm_positions = mlm_positions | ((torch.rand(input_ids.shape, device=input_ids.device) < math_mask_prob) & math_positions)

    labels = input_ids[mlm_positions]
    corrupted = input_ids.clone()
    choice = torch.rand(input_ids.shape, device=input_ids.device)
    mask_choice = mlm_positions & (choice < 0.80)
    random_choice = mlm_positions & (choice >= 0.80) & (choice < 0.90)
    corrupted[mask_choice] = cfg.mask_id
    random_tokens = torch.randint(5, cfg.vocab_size, input_ids.shape, device=input_ids.device)
    corrupted[random_choice] = random_tokens[random_choice]
    return corrupted, mlm_positions, labels


def apply_token_dropout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: TinyConfig,
    token_dropout: float,
) -> torch.Tensor:
    if token_dropout <= 0:
        return input_ids
    valid = attention_mask & (input_ids != cfg.pad_id) & (input_ids != cfg.cls_id) & (input_ids != cfg.sep_id)
    drop = (torch.rand(input_ids.shape, device=input_ids.device) < token_dropout) & valid
    out = input_ids.clone()
    out[drop] = cfg.mask_id
    return out


@torch.no_grad()
def evaluate_classifier(
    model: TinyMathBert,
    data: dict[str, torch.Tensor],
    batch_size: int,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    losses = []
    preds = []
    targets = []
    for batch in iter_device_batches(data, batch_size=batch_size, shuffle=False):
        logits = model.classify(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"], label_smoothing=label_smoothing)
        losses.append(float(loss.detach().cpu()))
        preds.append(torch.argmax(logits, dim=-1).detach().cpu())
        targets.append(batch["labels"].detach().cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_mlm(
    model: TinyMathBert,
    data: dict[str, torch.Tensor],
    cfg: TinyConfig,
    batch_size: int,
    max_batches: int,
    math_token_mask: torch.Tensor | None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    accuracies = []
    for i, batch in enumerate(iter_device_batches(data, batch_size=batch_size, shuffle=False)):
        if i >= max_batches:
            break
        corrupted, positions, labels = make_mlm_batch(
            batch["input_ids"], batch["attention_mask"], cfg, 0.15, 0.0, math_token_mask
        )
        hidden = model.encode(corrupted, batch["attention_mask"])
        logits = model.mlm_logits(hidden, positions)
        loss = F.cross_entropy(logits, labels)
        pred = torch.argmax(logits, dim=-1)
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float((pred == labels).float().mean().detach().cpu()))
    return {"mlm_loss": float(np.mean(losses)), "mlm_accuracy": float(np.mean(accuracies))}


def build_optimizer(model: TinyMathBert, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: TinyMathBert,
    cfg: TinyConfig,
    phase: str,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": cfg.to_dict(),
        "phase": phase,
        "epoch": epoch,
        "metrics": metrics,
        "params": count_parameters(model),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model(run_dir: Path, device: torch.device, checkpoint: Path | None = None) -> tuple[TinyMathBert, TinyConfig]:
    cfg = load_model_config(run_dir)
    model = TinyMathBert(cfg).to(device)
    params = assert_under_parameter_budget(model)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"], strict=True)
    return model, cfg


def load_math_token_mask(run_dir: Path, device: torch.device) -> torch.Tensor | None:
    path = run_dir / "data" / "math_token_mask.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(np.bool_)
    return torch.as_tensor(arr, dtype=torch.bool, device=device)


def train_phase(
    run_dir: Path,
    phase: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float = 0.05,
    mlm_prob: float = 0.15,
    mlm_weight: float = 0.5,
    token_dropout: float = 0.02,
    math_mask_prob: float = 0.05,
    eval_every: int = 1,
    save_every: int = 5,
    init_checkpoint: Path | None = None,
    device_name: str = "mps",
    amp: bool = False,
    grad_clip: float = 0.0,
    step_log_every: int = 20,
) -> dict[str, Any]:
    device = get_device(device_name)
    train = load_device_split(run_dir, "train", device)
    val = load_device_split(run_dir, "val", device)
    math_token_mask = load_math_token_mask(run_dir, device)
    model, cfg = load_model(run_dir, device=device, checkpoint=init_checkpoint)
    params = count_parameters(model)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(int(train["labels"].shape[0]) / batch_size)
    scheduler = build_scheduler(optimizer, total_steps=epochs * steps_per_epoch, warmup_steps=max(10, steps_per_epoch // 2))
    scaler_enabled = False
    log_path = run_dir / "logs" / f"{phase}.jsonl"
    best_score = float("-inf")
    best_path = run_dir / "checkpoints" / f"{phase}_best.pt"
    last_metrics: dict[str, Any] = {}

    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        n_seen = 0
        start = time.perf_counter()
        for step_idx, batch in enumerate(iter_device_batches(train, batch_size=batch_size, shuffle=True), start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type in {"mps", "cuda"}):
                if phase == "pretrain":
                    corrupted, positions, labels = make_mlm_batch(
                        batch["input_ids"], batch["attention_mask"], cfg, mlm_prob, math_mask_prob, math_token_mask
                    )
                    if labels.numel() == 0:
                        continue
                    hidden = model.encode(corrupted, batch["attention_mask"])
                    logits = model.mlm_logits(hidden, positions)
                    loss = F.cross_entropy(logits, labels)
                elif phase in {"finetune", "supervised"}:
                    ids = apply_token_dropout(batch["input_ids"], batch["attention_mask"], cfg, token_dropout)
                    logits = model.classify(ids, batch["attention_mask"])
                    loss = F.cross_entropy(logits, batch["labels"], label_smoothing=label_smoothing)
                elif phase in {"joint", "long_joint"}:
                    corrupted, positions, labels = make_mlm_batch(
                        batch["input_ids"], batch["attention_mask"], cfg, mlm_prob, math_mask_prob, math_token_mask
                    )
                    if labels.numel() == 0:
                        continue
                    hidden = model.encode(corrupted, batch["attention_mask"])
                    cls_logits = model.classify_from_hidden(hidden)
                    mlm_logits = model.mlm_logits(hidden, positions)
                    cls_loss = F.cross_entropy(cls_logits, batch["labels"], label_smoothing=label_smoothing)
                    mlm_loss = F.cross_entropy(mlm_logits, labels)
                    loss = cls_loss + mlm_weight * mlm_loss
                else:
                    raise ValueError(f"Unknown phase: {phase}")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            batch_n = int(batch["labels"].shape[0])
            epoch_loss = epoch_loss + loss.detach() * batch_n
            n_seen += batch_n
            if step_log_every > 0 and step_idx % step_log_every == 0:
                partial_elapsed = max(1e-9, time.perf_counter() - start)
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "step": step_idx,
                            "samples_seen": n_seen,
                            "samples_per_s": n_seen / partial_elapsed,
                            "tokens_per_s": n_seen * cfg.max_len / partial_elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        elapsed = max(1e-9, time.perf_counter() - start)
        samples_per_s = n_seen / elapsed
        tokens_per_s = n_seen * cfg.max_len / elapsed
        train_loss = float((epoch_loss / max(1, n_seen)).detach().cpu())
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "samples_per_s": samples_per_s,
            "tokens_per_s": tokens_per_s,
            "lr": optimizer.param_groups[0]["lr"],
            "params": params,
            "elapsed_s": time.perf_counter() - start_all,
            "memory": process_memory(),
            "mps_memory": mps_memory(),
        }
        if epoch % eval_every == 0:
            if phase == "pretrain":
                metrics.update(evaluate_mlm(model, val, cfg, batch_size=batch_size, max_batches=16, math_token_mask=math_token_mask))
                score = -metrics["mlm_loss"]
            else:
                val_metrics = evaluate_classifier(model, val, batch_size=batch_size, label_smoothing=0.0)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                score = metrics["val_macro_f1"]
            if score > best_score:
                best_score = score
                save_checkpoint(best_path, model, cfg, phase, epoch, metrics, optimizer=optimizer)
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(run_dir / "checkpoints" / f"{phase}_epoch_{epoch:04d}.pt", model, cfg, phase, epoch, metrics)
        save_checkpoint(run_dir / "checkpoints" / f"{phase}_last.pt", model, cfg, phase, epoch, metrics)
        append_jsonl(log_path, metrics)
        print(json.dumps(metrics, sort_keys=True))
        last_metrics = metrics
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite train loss in {phase} epoch {epoch}")

    return {"phase": phase, "best_score": best_score, "last_metrics": last_metrics, "best_checkpoint": str(best_path)}


def evaluate_checkpoint(run_dir: Path, checkpoint: Path, split: str, batch_size: int = 512, device_name: str = "mps") -> dict[str, Any]:
    device = get_device(device_name)
    data = load_device_split(run_dir, split, device)
    model, _ = load_model(run_dir, device=device, checkpoint=checkpoint)
    metrics = evaluate_classifier(model, data, batch_size=batch_size)
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in iter_device_batches(data, batch_size=batch_size, shuffle=False):
            logits = model.classify(batch["input_ids"], batch["attention_mask"])
            preds.append(torch.argmax(logits, dim=-1).detach().cpu())
            targets.append(batch["labels"].detach().cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    labels = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    result = {"split": split, "checkpoint": str(checkpoint), "metrics": metrics, "report": report, "confusion_matrix": cm}
    out_path = run_dir / "eval" / f"{checkpoint.stem}_{split}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def tiny_overfit_check(run_dir: Path, steps: int = 160, batch_size: int = 64, device_name: str = "mps") -> dict[str, Any]:
    device = get_device(device_name)
    train = load_device_split(run_dir, "train", device)
    subset = {k: v[:batch_size] for k, v in train.items()}
    model, cfg = load_model(run_dir, device=device)
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.0)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.classify(subset["input_ids"], subset["attention_mask"])
        loss = F.cross_entropy(logits, subset["labels"])
        loss.backward()
        optimizer.step()
    metrics = evaluate_classifier(model, subset, batch_size=batch_size)
    return {"steps": steps, "batch_size": batch_size, **metrics}
