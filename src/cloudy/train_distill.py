from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from cloudy.dataset import (
    BOS_TOKEN_ID,
    INST_START_ID,
    INST_END_ID,
    RESP_START_ID,
    TeacherDistillDataset,
)
from cloudy.model import CloudyConfig, CloudyForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cloudy.distill")


@dataclass
class DistillConfig:
    project_name: str = "Cloudy"
    experiment_name: str = "cloudy_distill_pilot"
    seed: int = 42

    # Model
    model: CloudyConfig = field(default_factory=CloudyConfig)

    # Data
    data_path: str = "experiments/cloudy_distill_v1/teacher_pilot_3.jsonl"
    tokenizer_path: str = "tokenizer/v2/tokenizer.json"

    # Training
    epochs: int = 15
    batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Paths
    output_dir: str = "experiments/cloudy_distill_pilot"
    device: str = "auto"


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def load_tokenizer_safe(tokenizer_path: str) -> Any:
    path = Path(tokenizer_path)
    if not path.is_file():
        # Check if directory contains tokenizer.json
        if path.is_dir() and (path / "tokenizer.json").is_file():
            path = path / "tokenizer.json"
        else:
            raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
    return Tokenizer.from_file(str(path))


def evaluate_loss(model: CloudyForCausalLM, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            if loss is not None and torch.isfinite(loss):
                total_loss += loss.item()
                count += 1
    return total_loss / max(1, count)


def run_distillation_pilot(config: DistillConfig, custom_tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """Execute the teacher-student distillation pilot."""
    set_seed(config.seed)
    device = get_device(config.device)
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    tokenizer = custom_tokenizer or load_tokenizer_safe(config.tokenizer_path)
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")

    # 2. Prepare Teacher Distillation Dataset
    dataset = TeacherDistillDataset(
        data_path_or_records=config.data_path,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        pad_to_max_len=True,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    stats = dataset.stats()
    logger.info(
        f"Prepared dataset: {stats['total_raw_examples']} teacher examples "
        f"expanded to {stats['total_training_chunks']} training chunks."
    )

    # 3. Initialize Fresh Random Student Model (Strictly no v1 weights)
    student_model = CloudyForCausalLM(config.model).to(device)
    num_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Initialized fresh student model with {num_params:,} parameters.")

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 4. Initial Loss Evaluation
    initial_loss = evaluate_loss(student_model, eval_dataloader, device)
    logger.info(f"Initial student loss before training: {initial_loss:.4f}")

    # 5. Training Loop
    step = 0
    step_logs: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        student_model.train()
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = student_model(input_ids, attention_mask=attention_mask, labels=labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at step {step}: {loss.item()}")

            loss.backward()

            # Verify and clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm=config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite grad norm at step {step}: {grad_norm.item()}")

            optimizer.step()

            step_logs.append({
                "epoch": epoch,
                "step": step,
                "loss": round(loss.item(), 4),
                "grad_norm": round(grad_norm.item(), 4),
            })

            logger.info(
                f"Epoch {epoch:02d}/{config.epochs:02d} | Step {step:03d} | "
                f"Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}"
            )

    elapsed_sec = time.time() - start_time
    final_loss = evaluate_loss(student_model, eval_dataloader, device)
    logger.info(f"Training completed in {elapsed_sec:.2f}s across {step} steps.")
    logger.info(f"Final student loss after training: {final_loss:.4f} (Reduction: {initial_loss - final_loss:.4f})")

    # 6. Save Checkpoint in Experiment Directory
    checkpoint_file = out_dir / f"cloudy_student_step{step}.pt"
    ckpt_payload = {
        "step": step,
        "model_state_dict": student_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config.model),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_time_seconds": elapsed_sec,
    }
    torch.save(ckpt_payload, checkpoint_file)
    logger.info(f"Saved student checkpoint: {checkpoint_file} ({checkpoint_file.stat().st_size / (1024**2):.2f} MiB)")

    # 7. Generation Smoke Test
    test_prompts = [
        "Explain what a variable is in Python using a simple example.",
        "Why should programmers write small, focused functions?",
    ]
    generations: List[Dict[str, str]] = []
    student_model.eval()

    logger.info("--- Generation Smoke Test ---")
    for prompt in test_prompts:
        prompt_tokens = (
            tokenizer.encode(prompt).ids
            if hasattr(tokenizer.encode(prompt), "ids")
            else tokenizer.encode(prompt)
        )
        input_seq = [BOS_TOKEN_ID, INST_START_ID] + list(prompt_tokens) + [INST_END_ID, RESP_START_ID]
        input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)

        gen_ids = student_model.generate(
            input_tensor,
            max_new_tokens=32,
            temperature=0.7,
            top_k=20,
        )[0].tolist()

        # Extract generated response portion after [RESP]
        resp_portion = gen_ids[len(input_seq):]
        try:
            gen_text = tokenizer.decode(resp_portion)
        except Exception:
            gen_text = f"<token_ids: {resp_portion}>"

        generations.append({"prompt": prompt, "generated": gen_text})
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated (post-distill student): {gen_text}")

    # 8. Save Metrics & Metadata
    metrics_file = out_dir / "distill_metrics.json"
    metrics_payload = {
        "project": config.project_name,
        "experiment": config.experiment_name,
        "total_steps": step,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "elapsed_seconds": round(elapsed_sec, 2),
        "checkpoint": str(checkpoint_file.resolve()),
        "generations": generations,
        "step_history": step_logs,
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info(f"Saved experiment metrics to: {metrics_file}")

    return metrics_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Cloudy Teacher-Student Distillation Pilot")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--data-path", type=str, default=None, help="Path to teacher .jsonl")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = DistillConfig()

    if args.config:
        cfg_path = Path(args.config)
        assert cfg_path.is_file(), f"Config file not found: {cfg_path}"
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if "model" in data:
                config.model = CloudyConfig(**data["model"])
            for key, val in data.items():
                if key != "model" and hasattr(config, key):
                    set_attr = getattr(config, key)
                    setattr(config, key, val)

    if args.data_path:
        config.data_path = args.data_path
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device

    run_distillation_pilot(config)


if __name__ == "__main__":
    main()
