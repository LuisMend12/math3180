from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_model_config, load_npz_split
from .model import TinyMathBert, count_parameters
from .torch_train import (
    build_optimizer,
    get_device,
    load_device_split,
    make_mlm_batch,
    process_memory,
    tiny_overfit_check,
)


def benchmark_torch(run_dir: Path, batch_sizes: list[int], steps: int, device_name: str = "mps", amp: bool = False) -> dict:
    device = get_device(device_name)
    cfg = load_model_config(run_dir)
    train = load_device_split(run_dir, "train", device)
    results = []
    for batch_size in batch_sizes:
        try:
            model = TinyMathBert(cfg).to(device)
            optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.03)
            torch.mps.empty_cache() if device.type == "mps" and hasattr(torch, "mps") else None
            # Warmup.
            for i, start in enumerate(range(0, batch_size * 2, batch_size)):
                batch = {k: v[start : start + batch_size] for k, v in train.items()}
                optimizer.zero_grad(set_to_none=True)
                corrupted, positions, labels = make_mlm_batch(batch["input_ids"], batch["attention_mask"], cfg, 0.15, 0.05, None)
                hidden = model.encode(corrupted, batch["attention_mask"])
                logits = model.mlm_logits(hidden, positions)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            start_time = time.perf_counter()
            n_seen = 0
            last_loss = 0.0
            for step in range(steps):
                offset = (step * batch_size) % (train["labels"].shape[0] - batch_size)
                batch = {k: v[offset : offset + batch_size] for k, v in train.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type in {"mps", "cuda"}):
                    corrupted, positions, labels = make_mlm_batch(batch["input_ids"], batch["attention_mask"], cfg, 0.15, 0.05, None)
                    hidden = model.encode(corrupted, batch["attention_mask"])
                    logits = model.mlm_logits(hidden, positions)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                n_seen += int(batch["labels"].shape[0])
                last_loss = float(loss.detach().cpu())
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start_time
            results.append(
                {
                    "batch_size": batch_size,
                    "ok": True,
                    "steps": steps,
                    "elapsed_s": elapsed,
                    "step_s": elapsed / steps,
                    "samples_per_s": n_seen / elapsed,
                    "tokens_per_s": n_seen * cfg.max_len / elapsed,
                    "last_loss": last_loss,
                }
            )
            print(json.dumps({"benchmark": "torch", "result": results[-1]}, sort_keys=True), flush=True)
        except Exception as exc:
            results.append({"batch_size": batch_size, "ok": False, "error": repr(exc)})
            print(json.dumps({"benchmark": "torch", "result": results[-1]}, sort_keys=True), flush=True)
    good = [r for r in results if r.get("ok")]
    best = max(good, key=lambda r: r["tokens_per_s"]) if good else None
    summary = {
        "backend": "torch",
        "device": str(device),
        "amp": amp,
        "params": count_parameters(TinyMathBert(cfg)),
        "results": results,
        "best": best,
        "memory": process_memory(),
    }
    out = run_dir / "benchmark_torch.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def benchmark_dataloader_workers(run_dir: Path, workers: list[int], batch_size: int, steps: int, device_name: str = "mps") -> dict:
    from torch.utils.data import DataLoader, TensorDataset

    cpu = load_npz_split(run_dir, "train")
    ds = TensorDataset(
        torch.as_tensor(cpu["input_ids"].astype(np.int64), dtype=torch.long),
        torch.as_tensor(cpu["attention_mask"].astype(np.bool_), dtype=torch.bool),
        torch.as_tensor(cpu["labels"], dtype=torch.long),
    )
    device = get_device(device_name)
    results = []
    for num_workers in workers:
        if sys.platform == "darwin" and num_workers > 0:
            result = {
                "num_workers": num_workers,
                "ok": False,
                "skipped": True,
                "reason": "macOS multiprocessing spawn is not used for the main path; device-resident batches are faster and avoid worker stalls",
            }
            results.append(result)
            print(json.dumps({"benchmark": "dataloader", "result": result}, sort_keys=True), flush=True)
            continue
        try:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            it = iter(loader)
            start = time.perf_counter()
            n_seen = 0
            for _ in range(steps):
                ids, mask, labels = next(it)
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                n_seen += int(labels.shape[0])
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            results.append({"num_workers": num_workers, "ok": True, "samples_per_s": n_seen / elapsed, "elapsed_s": elapsed})
            print(json.dumps({"benchmark": "dataloader", "result": results[-1]}, sort_keys=True), flush=True)
        except Exception as exc:
            results.append({"num_workers": num_workers, "ok": False, "error": repr(exc)})
            print(json.dumps({"benchmark": "dataloader", "result": results[-1]}, sort_keys=True), flush=True)
    summary = {"backend": "torch_dataloader", "batch_size": batch_size, "results": results}
    (run_dir / "benchmark_dataloader.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def benchmark_mlx(run_dir: Path, batch_size: int, steps: int) -> dict:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:
        return {"backend": "mlx", "ok": False, "error": repr(exc)}

    cfg = load_model_config(run_dir)
    data = load_npz_split(run_dir, "train")
    ids = mx.array(data["input_ids"].astype(np.int32))
    labels = mx.array(data["labels"].astype(np.int32))

    class MlxClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
            self.encoder = nn.TransformerEncoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.ff_dim, dropout=cfg.dropout)
            self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        def __call__(self, x):
            pos = mx.arange(x.shape[1])[None, :]
            h = self.token(x) + self.pos(pos)
            # MLX uses additive masks; for benchmark we use fixed-length cached inputs.
            h = self.encoder(h, None)
            return self.head(h[:, 0, :])

    model = MlxClassifier()
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.03)

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for step in range(2):
        offset = (step * batch_size) % (ids.shape[0] - batch_size)
        x = ids[offset : offset + batch_size]
        y = labels[offset : offset + batch_size]
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
    start = time.perf_counter()
    last_loss = None
    for step in range(steps):
        offset = (step * batch_size) % (ids.shape[0] - batch_size)
        x = ids[offset : offset + batch_size]
        y = labels[offset : offset + batch_size]
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        last_loss = float(loss.item())
    elapsed = time.perf_counter() - start
    summary = {
        "backend": "mlx",
        "ok": True,
        "batch_size": batch_size,
        "steps": steps,
        "elapsed_s": elapsed,
        "step_s": elapsed / steps,
        "samples_per_s": batch_size * steps / elapsed,
        "tokens_per_s": batch_size * steps * cfg.max_len / elapsed,
        "last_loss": last_loss,
    }
    (run_dir / "benchmark_mlx.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_full_benchmark(run_dir: Path, device_name: str = "mps") -> dict:
    torch_summary = benchmark_torch(run_dir, batch_sizes=[64, 128, 256, 384, 512], steps=8, device_name=device_name)
    best_bs = int(torch_summary["best"]["batch_size"]) if torch_summary.get("best") else 256
    loader_summary = benchmark_dataloader_workers(run_dir, workers=[0, 2, 4, 8], batch_size=min(best_bs, 512), steps=8, device_name=device_name)
    mlx_summary = benchmark_mlx(run_dir, batch_size=min(best_bs, 512), steps=10)
    overfit = tiny_overfit_check(run_dir, steps=120, batch_size=64, device_name=device_name)
    summary = {"torch": torch_summary, "dataloader": loader_summary, "mlx": mlx_summary, "tiny_overfit": overfit}
    (run_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
