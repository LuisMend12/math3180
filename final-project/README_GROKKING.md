# Tiny MathNet Grokking Encoder

This experiment trains a sub-1M parameter Transformer encoder from scratch on
MathNet only. It uses a train-split-only BPE tokenizer, masked-token
pretraining, supervised fine-tuning, optional joint training, and benchmark
gates for Apple Silicon.

## Commands

```bash
uv venv --python /Users/tanushv/.local/bin/python3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-grokking.txt

python -m grokking_encoder.cli prepare --data mathnet_dataset.json --run-dir runs/tiny_mathnet_grokking
python -m grokking_encoder.cli benchmark --run-dir runs/tiny_mathnet_grokking
python -m grokking_encoder.cli supervised --run-dir runs/tiny_mathnet_grokking --epochs 20 --batch-size 512
python -m grokking_encoder.cli pretrain --run-dir runs/tiny_mathnet_grokking --epochs 60 --batch-size 512
python -m grokking_encoder.cli finetune --run-dir runs/tiny_mathnet_grokking --init-checkpoint runs/tiny_mathnet_grokking/checkpoints/pretrain_best.pt --epochs 80 --batch-size 512
python -m grokking_encoder.cli joint --run-dir runs/tiny_mathnet_grokking --init-checkpoint runs/tiny_mathnet_grokking/checkpoints/finetune_best.pt --epochs 120 --batch-size 512
python -m grokking_encoder.cli evaluate --run-dir runs/tiny_mathnet_grokking --checkpoint runs/tiny_mathnet_grokking/checkpoints/joint_best.pt --split test
python -m grokking_encoder.cli report --run-dir runs/tiny_mathnet_grokking
```

The main training path keeps pre-tokenized arrays resident on the selected
device and computes MLM logits only for masked positions.
