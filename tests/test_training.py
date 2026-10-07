import json
import tempfile
from pathlib import Path

import pytest
import torch

from cloudy.checkpoint import load_checkpoint
from cloudy.model import CloudyConfig
from cloudy.tokenizer import MockCloudyTokenizer
from cloudy.train import TrainingConfig, train_distillation


def test_distillation_training_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Prepare simulated teacher dataset with 3 teacher demonstrations
        dataset_file = tmp_path / "pilot_teacher_3.jsonl"
        demonstrations = [
            {
                "id": "groq_pilot_001",
                "messages": [
                    {"role": "user", "content": "Explain what a variable is in Python using a simple example."},
                    {"role": "assistant", "content": "A variable is a named storage container for holding data."},
                ],
            },
            {
                "id": "groq_pilot_002",
                "messages": [
                    {"role": "user", "content": "Why should programmers write small, focused functions?"},
                    {"role": "assistant", "content": "Small functions are simpler to test, read, and maintain."},
                ],
            },
            {
                "id": "groq_pilot_003",
                "messages": [
                    {"role": "user", "content": "Explain the difference between a list and a tuple in Python."},
                    {"role": "assistant", "content": "Lists are mutable while tuples are fixed and immutable."},
                ],
            },
        ]
        with dataset_file.open("w", encoding="utf-8") as f:
            for item in demonstrations:
                f.write(json.dumps(item) + "\n")

        # 2. Configure training with fresh student weights
        config = TrainingConfig(
            project_name="CloudyTest",
            experiment_name="test_pilot",
            seed=42,
            model=CloudyConfig(
                vocab_size=1000,
                d_model=64,
                n_heads=2,
                n_layers=2,
                d_ff=128,
                max_seq_len=64,
                dropout=0.0,
            ),
            data_path=str(dataset_file),
            tokenizer_path="",
            learning_rate=1e-3,
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            epochs=3,
            batch_size=1,
            output_dir=str(tmp_path / "experiment_output"),
            device="cpu",
            checkpoint_interval=2,
        )

        tokenizer = MockCloudyTokenizer(vocab_size=1000)

        # 3. Execute training
        results = train_distillation(config, custom_tokenizer=tokenizer)

        # 4. Assert finite loss and reduction
        assert results["total_steps"] > 0
        assert results["initial_loss"] > 0
        assert results["final_loss"] > 0
        assert results["final_loss"] < results["initial_loss"], "Training loss must decrease after optimizer steps"

        # 5. Assert checkpoint exists and reloads
        ckpt_path = Path(results["checkpoint_path"])
        assert ckpt_path.is_file()

        loaded_ckpt = load_checkpoint(ckpt_path)
        assert "model_state_dict" in loaded_ckpt
        assert "optimizer_state_dict" in loaded_ckpt
        assert "config" in loaded_ckpt
        assert loaded_ckpt["step"] == results["total_steps"]

        # 6. Assert metrics log exists
        metrics_file = tmp_path / "experiment_output" / "logs" / "distill_run_metrics.json"
        assert metrics_file.is_file()

        # 7. Assert generation smoke test completed
        assert len(results["sample_generations"]) == 2
        for gen in results["sample_generations"]:
            assert "prompt" in gen
            assert "response" in gen
            assert isinstance(gen["response"], str)
