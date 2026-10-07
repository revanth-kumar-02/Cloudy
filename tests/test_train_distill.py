import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cloudy.dataset import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
from cloudy.model import CloudyConfig
from cloudy.train_distill import DistillConfig, run_distillation_pilot


class MockTokenizer:
    """Mock tokenizer for offline unit tests."""
    def get_vocab_size(self):
        return 1000

    def encode(self, text: str):
        tokens = [(hash(word) % 900) + 10 for word in text.split()]
        return SimpleNamespace(ids=tokens)

    def decode(self, token_ids):
        return f"decoded_{len(token_ids)}_tokens"


def test_distillation_pilot_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create a dummy teacher_pilot.jsonl
        data_file = tmp_path / "pilot.jsonl"
        records = [
            {
                "id": "pilot_001",
                "messages": [
                    {"role": "user", "content": "Explain what a variable is in Python."},
                    {"role": "assistant", "content": "A variable stores values in memory for easy reuse."},
                ],
            },
            {
                "id": "pilot_002",
                "messages": [
                    {"role": "user", "content": "Why write small functions?"},
                    {"role": "assistant", "content": "Small functions are easier to test and maintain."},
                ],
            },
            {
                "id": "pilot_003",
                "messages": [
                    {"role": "user", "content": "List vs tuple difference?"},
                    {"role": "assistant", "content": "Lists are mutable while tuples are immutable."},
                ],
            },
        ]
        with data_file.open("w", encoding="utf-8") as f:
            import json
            for r in records:
                f.write(json.dumps(r) + "\n")

        # 2. Setup tiny student config
        config = DistillConfig(
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
            data_path=str(data_file),
            tokenizer_path="",
            epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            output_dir=str(tmp_path / "output"),
            device="cpu",
        )

        tokenizer = MockTokenizer()

        # 3. Run distillation pilot
        result = run_distillation_pilot(config, custom_tokenizer=tokenizer)

        # 4. Verify results
        assert result["total_steps"] > 0
        assert result["initial_loss"] > 0
        assert result["final_loss"] > 0
        assert result["final_loss"] < result["initial_loss"], "Loss should decrease after optimizer steps"

        # Verify checkpoint file exists
        ckpt_path = Path(result["checkpoint"])
        assert ckpt_path.is_file()

        # Load checkpoint and verify content
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "config" in ckpt
        assert ckpt["step"] == result["total_steps"]

        # Verify metrics.json exists
        metrics_file = tmp_path / "output" / "distill_metrics.json"
        assert metrics_file.is_file()
