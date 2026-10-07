from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates model loss and perplexity over a dataloader."""
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)

        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            valid_batches += 1

    avg_loss = total_loss / max(1, valid_batches)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap exponent to prevent overflow

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "batches_evaluated": valid_batches,
    }
