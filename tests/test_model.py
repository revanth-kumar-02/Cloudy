import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cloudy.model import CloudyConfig, CloudyForCausalLM


@pytest.fixture
def tiny_config():
    return CloudyConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
    )


def test_model_instantiation_and_shapes(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids)
    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
    assert loss is None


def test_backward_and_finite_gradients(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels=labels)
    assert loss is not None
    assert torch.isfinite(loss).item()

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient non-finite for {name}"


def test_gradient_clipping(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = input_ids.clone()

    _, loss = model(input_ids, labels=labels)
    # Inflate loss to induce large grads
    (loss * 1000.0).backward()

    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    assert total_norm > max_norm

    # Re-calculate post-clipping norm
    post_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]),
        2,
    )
    assert post_norm <= max_norm + 1e-5


def test_optimizer_updates_weights(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Clone initial weight of first block
    init_weight = model.blocks[0].attn.qkv.weight.clone()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    labels = input_ids.clone()

    _, loss = model(input_ids, labels=labels)
    loss.backward()
    optimizer.step()

    updated_weight = model.blocks[0].attn.qkv.weight
    assert not torch.equal(init_weight, updated_weight)


def test_prompt_token_masking(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))

    # Test 1: all tokens supervised
    labels_all = input_ids.clone()
    _, loss_all = model(input_ids, labels=labels_all)

    # Test 2: prompt tokens masked with -100 (e.g. first 4 tokens masked)
    labels_masked = input_ids.clone()
    labels_masked[:, :4] = -100
    _, loss_masked = model(input_ids, labels=labels_masked)

    assert loss_masked is not None
    assert torch.isfinite(loss_masked).item()
    # Masked loss should be different because it only averages unmasked tokens
    assert loss_masked.item() != loss_all.item()


def test_checkpoint_save_and_load(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"

        payload = {
            "step": 42,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": tiny_config.__dict__,
        }
        torch.save(payload, ckpt_path)
        assert ckpt_path.is_file()

        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        new_model = CloudyForCausalLM(CloudyConfig(**loaded_ckpt["config"]))
        new_model.load_state_dict(loaded_ckpt["model_state_dict"])

        # Check weights match exactly
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


def test_generation_smoke_test(tiny_config):
    model = CloudyForCausalLM(tiny_config)
    prompt = torch.tensor([[10, 20, 30]])
    gen_tokens = model.generate(prompt, max_new_tokens=5, temperature=0.7, top_k=10)

    assert gen_tokens.shape == (1, 8)
    assert torch.equal(gen_tokens[:, :3], prompt)


def test_state_dict_keys_against_v1_architecture():
    """Verify that our reconstructed Cloudy module matches the exact 15 keys of v1 checkpoint."""
    config_v1 = CloudyConfig(
        vocab_size=32000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=128,
        dropout=0.0,
    )
    model = CloudyForCausalLM(config_v1)
    keys = list(model.state_dict().keys())
    assert len(keys) == 15
    expected_keys = [
        "token_embedding.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.out.weight",
        "blocks.0.norm2.weight",
        "blocks.0.ff.0.weight",
        "blocks.0.ff.2.weight",
        "blocks.1.norm1.weight",
        "blocks.1.attn.qkv.weight",
        "blocks.1.attn.out.weight",
        "blocks.1.norm2.weight",
        "blocks.1.ff.0.weight",
        "blocks.1.ff.2.weight",
        "norm.weight",
        "lm_head.weight",
    ]
    assert keys == expected_keys
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 8585856
