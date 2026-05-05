from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_full_benchmark
from .config import default_run_dir
from .data import prepare_dataset
from .report import build_report
from .torch_train import evaluate_checkpoint, train_phase


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--mlm-weight", type=float, default=0.5)
    parser.add_argument("--token-dropout", type=float, default=0.02)
    parser.add_argument("--math-mask-prob", type=float, default=0.05)
    parser.add_argument("--eval-every", type=positive_int, default=1)
    parser.add_argument("--save-every", type=positive_int, default=5)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--step-log-every", type=int, default=20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny MathNet grokking encoder CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--data", type=Path, default=Path("mathnet_dataset.json"))
    p_prepare.add_argument("--run-dir", type=Path, default=default_run_dir())
    p_prepare.add_argument("--vocab-size", type=positive_int, default=4096)
    p_prepare.add_argument("--max-len", type=positive_int, default=256)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--group-duplicates", action="store_true")

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--run-dir", type=Path, default=default_run_dir())
    p_bench.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")

    p_supervised = sub.add_parser("supervised")
    add_train_args(p_supervised)
    p_supervised.set_defaults(phase="supervised")

    p_pretrain = sub.add_parser("pretrain")
    add_train_args(p_pretrain)
    p_pretrain.set_defaults(phase="pretrain", label_smoothing=0.0)

    p_finetune = sub.add_parser("finetune")
    add_train_args(p_finetune)
    p_finetune.set_defaults(phase="finetune")

    p_joint = sub.add_parser("joint")
    add_train_args(p_joint)
    p_joint.set_defaults(phase="joint")

    p_long_joint = sub.add_parser("long-joint")
    add_train_args(p_long_joint)
    p_long_joint.set_defaults(phase="long_joint")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run-dir", type=Path, default=default_run_dir())
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_eval.add_argument("--batch-size", type=positive_int, default=512)
    p_eval.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")

    p_report = sub.add_parser("report")
    p_report.add_argument("--run-dir", type=Path, default=default_run_dir())

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = prepare_dataset(args.data, args.run_dir, args.vocab_size, args.max_len, args.seed, args.group_duplicates)
    elif args.cmd == "benchmark":
        result = run_full_benchmark(args.run_dir, device_name=args.device)
    elif args.cmd in {"supervised", "pretrain", "finetune", "joint", "long-joint"}:
        result = train_phase(
            run_dir=args.run_dir,
            phase=args.phase,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            mlm_prob=args.mlm_prob,
            mlm_weight=args.mlm_weight,
            token_dropout=args.token_dropout,
            math_mask_prob=args.math_mask_prob,
            eval_every=args.eval_every,
            save_every=args.save_every,
            init_checkpoint=args.init_checkpoint,
            device_name=args.device,
            amp=args.amp,
            grad_clip=args.grad_clip,
            step_log_every=args.step_log_every,
        )
    elif args.cmd == "evaluate":
        result = evaluate_checkpoint(args.run_dir, args.checkpoint, args.split, args.batch_size, args.device)
    elif args.cmd == "report":
        result = build_report(args.run_dir)
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
