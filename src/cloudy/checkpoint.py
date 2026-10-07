from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("cloudy.checkpoint")


def get_rng_states() -> Dict[str, Any]:
    states = {
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]):
    if "python_rng_state" in states:
        random.setstate(states["python_rng_state"])
    if "torch_rng_state" in states:
        torch.set_rng_state(states["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in states:
        torch.cuda.set_rng_state_all(states["cuda_rng_state"])


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> Path:
    """Atomically saves model checkpoint with full reproducibility metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or f"cloudy_student_step_{step}.pt"
    final_path = checkpoint_dir / fname
    tmp_path = checkpoint_dir / f"{fname}.tmp"

    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "rng_states": get_rng_states(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics

    # Atomic write pattern
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)

    logger.info(f"Saved atomic checkpoint to: {final_path} ({final_path.stat().st_size / (1024**2):.2f} MiB)")
    return final_path


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Loads checkpoint safely mapping to the specified device."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    ckpt = torch.load(p, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from: {p} (step: {ckpt.get('step', 'unknown')})")
    return ckpt
