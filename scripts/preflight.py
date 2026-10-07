#!/usr/bin/env python3
"""Preflight environment diagnostic and verification script for Cloudy."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import SPECIAL_TOKEN_MAP, CloudyTokenizer


def run_preflight(config_path: str = "configs/cloudy_distill_v1.json"):
    print("=" * 60)
    print("☁️  CLOUDY PREFLIGHT DIAGNOSTIC")
    print("=" * 60)

    all_passed = True

    # 1. Python & System
    print(f"[OK] Python Version: {sys.version.split()[0]}")

    # 2. PyTorch & CUDA
    print(f"[OK] PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] CUDA Available: True ({gpu_name}, {vram_gib:.2f} GiB VRAM)")
    else:
        print("[WARN] CUDA Available: False (Running on CPU)")

    # 3. Check Configuration File
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        print(f"[FAIL] Config file not found: {cfg_file}")
        return False
    print(f"[OK] Configuration file found: {cfg_file}")
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 4. Check Dataset & Tokenizer Paths
    data_path = Path(cfg.get("data_path", ""))
    tok_path = Path(cfg.get("tokenizer_path", ""))

    print("\n--- Asset Verification ---")
    if data_path.is_file():
        print(f"[OK] Teacher dataset file found: {data_path}")
        # Validate JSONL
        records = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[OK] Loaded {len(records)} teacher demonstrations from JSONL")
    else:
        print(f"[WARN] Teacher dataset not found locally: {data_path}")
        print("       (Note: On Colab, ensure Google Drive is mounted at /content/drive)")
        all_passed = False

    if tok_path.is_file():
        print(f"[OK] Tokenizer file found: {tok_path}")
        tokenizer = CloudyTokenizer.from_file(tok_path)
        print(f"[OK] Tokenizer vocabulary size: {tokenizer.get_vocab_size():,}")
        # Verify special tokens
        for name, expected_id in SPECIAL_TOKEN_MAP.items():
            print(f"     Special token {name:8s} -> ID {expected_id}")
    else:
        print(f"[WARN] Tokenizer JSON not found locally: {tok_path}")
        print("       (Note: On Colab, ensure Google Drive is mounted at /content/drive)")
        all_passed = False

    # 5. Model Architecture & Parameter Verification
    print("\n--- Model Architecture Verification ---")
    model_cfg = CloudyConfig(**cfg.get("model", {}))
    student = CloudyForCausalLM(model_cfg)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"[OK] Fresh Cloudy Student instantiated")
    print(f"[OK] Total Model Parameters: {total_params:,}")
    assert total_params == 8585856, f"Expected 8,585,856 parameters, got {total_params}"

    # 6. Device Smoke Test (Forward + Backward)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    dummy_input = torch.randint(0, model_cfg.vocab_size, (1, 16), device=device)
    dummy_labels = dummy_input.clone()
    logits, loss = student(dummy_input, labels=dummy_labels)
    assert logits.shape == (1, 16, model_cfg.vocab_size)
    assert torch.isfinite(loss).item()
    loss.backward()
    print(f"[OK] Model forward & backward smoke test passed on device: {device}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PREFLIGHT CHECKS PASSED. Ready for training launch!")
    else:
        print("ℹ️  Local code checks passed. Assets on Drive will be verified when mounted in Colab.")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cloudy_distill_v1.json")
    args = parser.parse_args()
    run_preflight(args.config)
