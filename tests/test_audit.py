import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloudy.checkpoint import save_checkpoint
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import MockCloudyTokenizer
from scripts.audit_checkpoint import audit_checkpoint


def test_audit_checkpoint_deterministic_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Dummy dataset
        data_file = tmp / "pilot.jsonl"
        records = [
            {
                "id": "p1",
                "messages": [
                    {"role": "user", "content": "Explain what a variable is in Python using a simple example."},
                    {"role": "assistant", "content": "A variable is a labeled memory container."},
                ],
            },
            {
                "id": "p2",
                "messages": [
                    {"role": "user", "content": "Why should programmers write small, focused functions?"},
                    {"role": "assistant", "content": "Small functions are easy to test and debug."},
                ],
            },
            {
                "id": "p3",
                "messages": [
                    {"role": "user", "content": "Explain the difference between a list and a tuple in Python."},
                    {"role": "assistant", "content": "Lists are mutable while tuples are immutable."},
                ],
            },
        ]
        with data_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # 2. Dummy tokenizer json
        tok_file = tmp / "tokenizer.json"
        tok_file.write_text("{}", encoding="utf-8")

        # 3. Dummy trained model checkpoint
        cfg = CloudyConfig(vocab_size=1000, d_model=64, n_heads=2, n_layers=2, d_ff=128, max_seq_len=64)
        model = CloudyForCausalLM(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_file = save_checkpoint(
            checkpoint_dir=tmp,
            step=100,
            model=model,
            optimizer=opt,
            config=cfg.__dict__,
            metrics={"final_loss": 4.82, "perplexity": 124.0},
            filename="final_pilot.pt",
        )

        # Monkeypatch CloudyTokenizer.from_file to return MockCloudyTokenizer
        from cloudy.tokenizer import CloudyTokenizer

        orig_from_file = CloudyTokenizer.from_file
        CloudyTokenizer.from_file = classmethod(lambda cls, path: MockCloudyTokenizer(vocab_size=1000))

        try:
            success = audit_checkpoint(
                checkpoint_path=str(ckpt_file),
                tokenizer_path=str(tok_file),
                data_path=str(data_file),
                device_name="cpu",
                seed=42,
            )
            assert success is True
        finally:
            CloudyTokenizer.from_file = orig_from_file
