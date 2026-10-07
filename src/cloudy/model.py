from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CloudyConfig:
    vocab_size: int = 32000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.0
    eps: float = 1e-6
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE)."""
    assert dim % 2 == 0, f"Dimension {dim} must be divisible by 2 for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [max_seq_len, dim // 2]


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to query or key tensor.
    
    Args:
        x: Tensor of shape [batch_size, n_heads, seq_len, head_dim]
        freqs_cis: Complex tensor of shape [seq_len, head_dim // 2]
    """
    batch_size, n_heads, seq_len, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(batch_size, n_heads, seq_len, -1, 2))
    freqs_cis = freqs_cis[:seq_len].view(1, 1, seq_len, -1)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)


class CloudyAttention(nn.Module):
    """Causal Multi-Head Self-Attention with merged QKV projection and RoPE."""

    def __init__(self, config: CloudyConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by n_heads"

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project merged QKV
        qkv = self.qkv(x)  # [batch, seq_len, 3 * d_model]
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Reshape to [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Scaled dot-product attention with causal mask
        # If attention_mask is given (e.g. [batch, 1, 1, seq_len] for padding), merge with causal
        is_causal = attention_mask is None
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask if not is_causal else None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Transpose back and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out(attn_out)


class CloudyBlock(nn.Module):
    """Transformer decoder block with RMSNorm, Self-Attention, and MLP."""

    def __init__(self, config: CloudyConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.eps)
        self.attn = CloudyAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.eps)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class CloudyForCausalLM(nn.Module):
    """Cloudy Decoder-Only Language Model."""

    def __init__(self, config: CloudyConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([CloudyBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Register precomputed RoPE frequencies as a persistent buffer
        head_dim = config.d_model // config.n_heads
        freqs_cis = precompute_rope_freqs_cis(head_dim, config.max_seq_len, config.rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights with standard normal distribution scaled by fan_in
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid token, 0 for padding)
            labels: [batch_size, seq_len] (-100 for ignored tokens)
            
        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, (
            f"Input sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        h = self.token_embedding(input_ids)
        freqs_cis = self.freqs_cis[:seq_len]

        # Prepare causal attention mask with padding mask support if attention_mask provided
        expanded_mask = None
        if attention_mask is not None:
            # Create [batch, 1, seq_len, seq_len] causal + padding mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            expanded_mask = pad_mask | causal_mask.unsqueeze(0).unsqueeze(0)
            expanded_mask = expanded_mask.to(dtype=torch.bool)
            # For PyTorch SDPA, convert bool mask: True means masked out -> float -inf
            attn_mask_float = torch.zeros(batch_size, 1, seq_len, seq_len, device=input_ids.device, dtype=h.dtype)
            attn_mask_float = attn_mask_float.masked_fill(expanded_mask, float("-inf"))
            expanded_mask = attn_mask_float

        for block in self.blocks:
            h = block(h, freqs_cis, attention_mask=expanded_mask)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        eos_token_id: Optional[int] = 1,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        curr_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= self.config.max_seq_len:
                break

            logits, _ = self.forward(curr_ids)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0 or temperature < 1e-5:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    top_k_val = min(top_k, scaled_logits.size(-1))
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k_val)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = float("-inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return curr_ids
