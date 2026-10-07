from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cloudy.checkpoint import save_checkpoint
from cloudy.data import TeacherDistillationDataset, create_distillation_dataloader
from cloudy.evaluation import evaluate_model
from cloudy.generate import generate_response
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import CloudyTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cloudy.train")


@dataclass
class TrainingConfig:
    project_name: str = "Cloudy"
    experiment_name: str = "cloudy_distill_v1"
    seed: int = 42

    # Model architecture
    model: CloudyConfig = field(default_factory=CloudyConfig)

    # Data paths
    data_path: str = "/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/teacher_pilot_3.jsonl"
    tokenizer_path: str = "/content/drive/MyDrive/scratch_llm_1b/tokenizer/v2/tokenizer.json"

    # Optimization parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    epochs: int = 20
    batch_size: int = 1

    # Output & Device
    output_dir: str = "/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/pilot_run"
    device: str = "auto"
    eval_interval: int = 5
    checkpoint_interval: int = 10


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return 1.0
    return float(step) / float(max(1, warmup_steps))


def train_distillation(
    config: TrainingConfig,
    custom_tokenizer: Optional[CloudyTokenizer] = None,
) -> Dict[str, Any]:
    """Runs the teacher-student distillation training loop."""
    set_seed(config.seed)
    device = get_device(config.device)
    logger.info(f"Target execution device: {device} (CUDA: {torch.cuda.is_available()})")

    out_dir = Path(config.output_dir)
    checkpoints_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    if custom_tokenizer is not None:
        tokenizer = custom_tokenizer
    else:
        tokenizer = CloudyTokenizer.from_file(config.tokenizer_path)
    logger.info(f"Loaded tokenizer with vocabulary size: {tokenizer.get_vocab_size():,}")

    # 2. Load Dataset
    dataset = TeacherDistillationDataset(
        data_path_or_records=config.data_path,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        pad_to_max_len=True,
    )
    summary = dataset.get_summary()
    logger.info(
        f"Loaded {summary['total_examples']} teacher demonstrations "
        f"({summary['total_training_chunks']} training chunks, {summary['overflow_examples']} overflow handled)."
    )

    dataloader = create_distillation_dataloader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = create_distillation_dataloader(dataset, batch_size=config.batch_size, shuffle=False)

    # 3. Fresh Student Initialization (Strictly fresh random weights)
    student = CloudyForCausalLM(config.model).to(device)
    param_count = sum(p.numel() for p in student.parameters())
    logger.info(f"Initialized fresh Cloudy student with {param_count:,} parameters.")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Initial loss measurement
    pre_eval = evaluate_model(student, eval_loader, device)
    initial_loss = pre_eval["eval_loss"]
    logger.info(f"Initial baseline loss (pre-training): {initial_loss:.4f} (Perplexity: {pre_eval['perplexity']:.2f})")

    step = 0
    accum_steps = max(1, config.gradient_accumulation_steps)
    step_history: List[Dict[str, Any]] = []
    start_time = time.time()

    # 4. Training Loop
    optimizer.zero_grad()
    for epoch in range(1, config.epochs + 1):
        student.train()
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = student(input_ids, attention_mask=attention_mask, labels=labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite loss at step {step + 1}: {loss.item()}")

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                step += 1

                # Warmup schedule adjustment
                lr_mult = get_lr_multiplier(step, config.warmup_steps)
                current_lr = config.learning_rate * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                # Check and clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=config.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Encountered non-finite grad norm at step {step}: {grad_norm.item()}")

                optimizer.step()
                optimizer.zero_grad()

                step_history.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "grad_norm": round(grad_norm.item(), 4),
                    "lr": current_lr,
                })

                logger.info(
                    f"Epoch {epoch:02d}/{config.epochs:02d} | Step {step:03d} | "
                    f"Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f} | LR: {current_lr:.6f}"
                )

                if step % config.checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_dir=checkpoints_dir,
                        step=step,
                        model=student,
                        optimizer=optimizer,
                        config=asdict(config.model),
                    )

    elapsed = time.time() - start_time
    post_eval = evaluate_model(student, eval_loader, device)
    final_loss = post_eval["eval_loss"]
    logger.info(f"Training completed in {elapsed:.2f}s.")
    logger.info(f"Final student loss: {final_loss:.4f} (Reduction: {initial_loss - final_loss:.4f})")

    # 5. Final Checkpoint
    final_ckpt = save_checkpoint(
        checkpoint_dir=checkpoints_dir,
        step=step,
        model=student,
        optimizer=optimizer,
        config=asdict(config.model),
        metrics={"initial_loss": initial_loss, "final_loss": final_loss, "perplexity": post_eval["perplexity"]},
        filename="cloudy_student_distill_final.pt",
    )

    # 6. Generation Smoke Test
    test_prompts = [
        "Explain what a variable is in Python using a simple example.",
        "Why should programmers write small, focused functions?",
    ]
    sample_generations: List[Dict[str, str]] = []
    logger.info("--- Generation Evaluation ---")
    for pr in test_prompts:
        response_text = generate_response(student, tokenizer, pr, max_new_tokens=32, device=device)
        sample_generations.append({"prompt": pr, "response": response_text})
        logger.info(f"Prompt: {pr}")
        logger.info(f"Response: {response_text}")

    # 7. Write Run Metrics Log
    metrics_log_path = logs_dir / "distill_run_metrics.json"
    run_summary = {
        "project": config.project_name,
        "experiment": config.experiment_name,
        "total_steps": step,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_reduction": round(initial_loss - final_loss, 4),
        "final_perplexity": round(post_eval["perplexity"], 2),
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint_path": str(final_ckpt.resolve()),
        "sample_generations": sample_generations,
        "step_history": step_history,
    }
    metrics_log_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    logger.info(f"Metrics saved to: {metrics_log_path}")

    return run_summary


def load_config_from_json(json_path: Path) -> TrainingConfig:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model_cfg = CloudyConfig(**data.get("model", {})) if "model" in data else CloudyConfig()
    cfg_kwargs = {k: v for k, v in data.items() if k != "model"}
    return TrainingConfig(model=model_cfg, **cfg_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Cloudy Teacher-Student Distillation Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    assert config_path.is_file(), f"Config file not found: {config_path}"

    config = load_config_from_json(config_path)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device

    train_distillation(config)


if __name__ == "__main__":
    main()
