#!/usr/bin/env python3
"""Deterministic evaluation and audit script for Cloudy pilot checkpoint.

Loads the existing trained student checkpoint from Google Drive, computes loss and
perplexity on the 3 teacher demonstrations without retraining, and generates
deterministic responses for all 3 prompts under both greedy and sampled decoding.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from cloudy.checkpoint import load_checkpoint
from cloudy.data import TeacherDistillationDataset, create_distillation_dataloader
from cloudy.evaluation import evaluate_model
from cloudy.generate import generate_response
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import CloudyTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def audit_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    data_path: str,
    device_name: str = "auto",
    seed: int = 42,
):
    set_seed(seed)
    print("=" * 65)
    print("🔍 CLOUDY PILOT CHECKPOINT AUDIT")
    print("=" * 65)

    # 1. Device Selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 2. Check File Existence
    ckpt_file = Path(checkpoint_path)
    tok_file = Path(tokenizer_path)
    data_file = Path(data_path)

    for name, p in [("Checkpoint", ckpt_file), ("Tokenizer", tok_file), ("Dataset", data_file)]:
        if not p.is_file():
            print(f"[FAIL] {name} not found: {p}")
            return False
        print(f"[OK] {name} found: {p}")

    # 3. Load Tokenizer
    tokenizer = CloudyTokenizer.from_file(tok_file)
    print(f"[OK] Tokenizer loaded (vocab size: {tokenizer.get_vocab_size():,})")

    # 4. Load Checkpoint Safely
    ckpt = load_checkpoint(ckpt_file, device=device)
    saved_step = ckpt.get("step", "unknown")
    saved_config = ckpt.get("config", {})
    metrics = ckpt.get("metrics", {})
    print(f"[OK] Checkpoint loaded successfully:")
    print(f"     Step: {saved_step}")
    print(f"     Recorded Training Loss: {metrics.get('final_loss', 'N/A')}")
    print(f"     Recorded Perplexity:    {metrics.get('perplexity', 'N/A')}")

    # 5. Restore Student Model
    model_cfg = CloudyConfig(**saved_config) if saved_config else CloudyConfig()
    model = CloudyForCausalLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Restored Cloudy student ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # 6. Evaluate Loss on the 3 Teacher Demonstrations
    dataset = TeacherDistillationDataset(
        data_path_or_records=data_file,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        pad_to_max_len=True,
    )
    dataloader = create_distillation_dataloader(dataset, batch_size=1, shuffle=False)
    summary = dataset.get_summary()

    eval_results = evaluate_model(model, dataloader, device)
    print("\n--- Quantitative Audit on Teacher Dataset ---")
    print(f"Demonstrations:     {summary['total_examples']}")
    print(f"Training Chunks:    {summary['total_training_chunks']}")
    print(f"Audited Loss:       {eval_results['eval_loss']:.4f}")
    print(f"Audited Perplexity: {eval_results['perplexity']:.2f}")

    # 7. Deterministic Generation for All 3 Prompts
    prompts = [
        "Explain what a variable is in Python using a simple example.",
        "Why should programmers write small, focused functions?",
        "Explain the difference between a list and a tuple in Python.",
    ]

    print("\n--- Deterministic Generation Audit (Seed: 42) ---")
    for idx, pr in enumerate(prompts, start=1):
        print(f"\n[Prompt {idx}]: {pr}")

        # Greedy Decoding (temperature=0.0)
        set_seed(seed)
        greedy_resp = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=pr,
            max_new_tokens=48,
            temperature=0.0,
            device=device,
        )
        print(f"  Greedy (temp=0.0):")
        print(f"    {greedy_resp if greedy_resp else '<EMPTY_OUTPUT>'}")

        # Sampled Decoding (temperature=0.7, top_k=20)
        set_seed(seed)
        sampled_resp = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=pr,
            max_new_tokens=48,
            temperature=0.7,
            top_k=20,
            device=device,
        )
        print(f"  Sampled (temp=0.7, top_k=20):")
        print(f"    {sampled_resp if sampled_resp else '<EMPTY_OUTPUT>'}")

    print("\n" + "=" * 65)
    print("✅ AUDIT COMPLETE — No files or checkpoints were modified.")
    print("=" * 65 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Audit Cloudy Pilot Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/pilot_run/checkpoints/cloudy_student_distill_final.pt",
        help="Path to final student checkpoint .pt file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/tokenizer/v2/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/content/drive/MyDrive/scratch_llm_1b/experiments/cloudy_distill_v1/teacher_pilot_3.jsonl",
        help="Path to teacher_pilot_3.jsonl",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    args = parser.parse_args()

    audit_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
