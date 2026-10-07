#!/usr/bin/env python3
"""One-command training runner on Google Colab Tesla T4 GPU."""

import argparse
import os
import sys
from pathlib import Path

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from cloudy.train import load_config_from_json, train_distillation
from scripts.preflight import run_preflight
from scripts.verify_data_pipeline import run_pipeline_verification


def main():
    parser = argparse.ArgumentParser(description="Cloudy Colab Training Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cloudy_distill_fixed_chunking.json",
        help="Path to JSON training configuration",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # 1. Mount Drive if on Colab and not already mounted
    if "google.colab" in sys.modules or os.path.exists("/content"):
        drive_path = Path("/content/drive/MyDrive")
        if not drive_path.exists():
            print("Mounting Google Drive at /content/drive...")
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception as e:
                print(f"[WARN] Colab drive mount returned: {e}")

    # 2. Run Preflight
    print("\nRunning preflight checks...")
    run_preflight(args.config)

    # 3. Load Training Config
    cfg_path = Path(args.config)
    assert cfg_path.is_file(), f"Configuration file not found: {cfg_path}"
    config = load_config_from_json(cfg_path)

    # 4. Mandatory Pre-Training Data Pipeline Verification
    print("\nRunning mandatory data pipeline & chunking verification...")
    pipeline_ok = run_pipeline_verification(
        data_path=config.data_path,
        tokenizer_path=config.tokenizer_path,
        max_seq_len=config.model.max_seq_len,
    )
    if not pipeline_ok:
        print("[FAIL] Preflight pipeline verification failed. Aborting training to protect artifacts.")
        sys.exit(1)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr

    # 5. Launch Isolated Training Experiment
    print(f"\nStarting Cloudy Fresh Distillation Experiment ({config.experiment_name}) on Tesla T4...")
    print(f"Checkpoints will be saved exclusively to: {config.output_dir}")
    metrics = train_distillation(config)

    print("\n" + "=" * 60)
    print("🎉 FRESH ISOLATED TRAINING EXPERIMENT COMPLETED!")
    print(f"Total Steps:              {metrics['total_steps']}")
    print(f"Supervised Tokens:        {metrics.get('total_supervised_tokens', 'N/A')}")
    print(f"Initial Loss:             {metrics['initial_loss']:.4f}")
    print(f"Final Loss:               {metrics['final_loss']:.4f}")
    print(f"Loss Reduction:           {metrics['loss_reduction']:.4f}")
    print(f"Final Perplexity:         {metrics['final_perplexity']:.2f}")
    print(f"New Checkpoint Saved:     {metrics['checkpoint_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
