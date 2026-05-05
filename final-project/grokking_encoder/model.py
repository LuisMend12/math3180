from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TinyConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        key_mask = attention_mask[:, None, None, :].to(torch.bool)
        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        except Exception:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min / 2)
            probs = torch.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = x + self.drop1(self.attn(self.norm1(x), attention_mask))
        x = x * mask
        x = x + self.drop2(self.ffn(self.norm2(x)))
        x = x * mask
        return x


class TinyMathEncoder(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)


class TinyMathBert(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TinyMathEncoder(cfg)
        self.cls_drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)
        self.mlm_dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlm_norm = nn.LayerNorm(cfg.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask)

    def classify_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        cls = hidden[:, 0]
        return self.classifier(self.cls_drop(cls))

    def classify(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)
        return self.classify_from_hidden(hidden)

    def mlm_logits(self, hidden: torch.Tensor, masked_positions: torch.Tensor | None = None) -> torch.Tensor:
        if masked_positions is not None:
            hidden = hidden[masked_positions]
        hidden = self.mlm_norm(F.gelu(self.mlm_dense(hidden)))
        return F.linear(hidden, self.encoder.token_emb.weight, self.mlm_bias)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_under_parameter_budget(model: nn.Module, budget: int = 1_000_000) -> int:
    params = count_parameters(model)
    if params >= budget:
        raise ValueError(f"Model has {params:,} parameters, above budget {budget:,}")
    return params
