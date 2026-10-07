import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cloudy.data import (
    RESP_START_ID,
    TeacherDistillationDataset,
    create_distillation_dataloader,
    format_and_chunk_example,
)
from cloudy.generate import format_inference_prompt, generate_response
from cloudy.model import CloudyConfig, CloudyForCausalLM
from cloudy.tokenizer import CloudyTokenizer, MockCloudyTokenizer


def get_trained_mock_bpe_tokenizer():
    raw_tok = Tokenizer(models.BPE())
    raw_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "[INST]", "[/INST]", "[RESP]", "[/RESP]"],
    )
    corpus = [
        "Why should programmers write small, focused functions?",
        "Small functions are easier to test, read, and maintain. They do one thing well, which reduces bugs.",
        "Explain what a variable is in Python using a simple example.",
        "A variable is a named storage container for holding data in memory.",
    ]
    raw_tok.train_from_iterator(corpus, trainer)
    return CloudyTokenizer(raw_tok)


def test_no_conflicting_targets_in_continuation_chunks():
    """Verify that continuation chunks never attach [RESP] to middle-of-response slices."""
    tok = get_trained_mock_bpe_tokenizer()
    prompt = "Explain what a variable is in Python using a simple example."
    # Long response exceeding 64 tokens
    response = "Small functions are easier to test, read, and maintain. " * 8

    chunks, stats = format_and_chunk_example("ex1", prompt, response, tok, max_seq_len=64)
    assert len(chunks) > 1, "Should generate multiple chunks for overflow test"

    # Chunk 1: [RESP] must predict the first token of the response
    c1 = chunks[0]
    resp_start_idx = (c1.input_ids == RESP_START_ID).nonzero(as_tuple=True)[0].item()
    c1_first_target = c1.labels[resp_start_idx + 1].item()
    assert c1_first_target != -100

    # Chunks 2+: [RESP] must NOT exist, preventing conflicting targets
    for idx, c in enumerate(chunks[1:], start=2):
        has_resp = (c.input_ids == RESP_START_ID).any().item()
        assert not has_resp, f"Chunk {idx} contains [RESP], creating conflicting targets!"


def test_single_demo_overfitting_verbatim_reproduction():
    """Regression test proving the model architecture and training pipeline

    can overfit a held-out demonstration and reproduce it verbatim.
    """
    tok = get_trained_mock_bpe_tokenizer()
    prompt = "Why should programmers write small, focused functions?"
    target_answer = "Small functions are easier to test, read, and maintain. They do one thing well, which reduces bugs."

    demo = [{
        "id": "single_demo",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_answer},
        ],
    }]

    dataset = TeacherDistillationDataset(demo, tokenizer=tok, max_seq_len=128)
    loader = create_distillation_dataloader(dataset, batch_size=1, shuffle=False)
    batch = list(loader)[0]

    cfg = CloudyConfig(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128)
    model = CloudyForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for 90 steps on the single demonstration
    for step in range(1, 91):
        optimizer.zero_grad()
        _, loss = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"Loss should drop below 0.1, got {loss.item():.4f}"

    # Generate greedy response
    model.eval()
    generated = generate_response(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        max_new_tokens=64,
        temperature=0.0,
    )

    assert generated == target_answer, (
        f"Generated text does not match target!\nTarget:    '{target_answer}'\nGenerated: '{generated}'"
    )


def test_top10_next_token_probabilities_and_loss_accounting():
    """Verify top-10 probability calculation and token-level loss accounting."""
    tok = get_trained_mock_bpe_tokenizer()
    prompt = "Explain what a variable is in Python using a simple example."
    answer = "A variable is a named storage container for holding data in memory."

    demo = [{
        "id": "demo_tokens",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }]

    dataset = TeacherDistillationDataset(demo, tokenizer=tok, max_seq_len=128)
    chunk = dataset[0]

    # Verify exact supervised response token count
    supervised_tokens = (chunk["labels"] != -100).sum().item()
    resp_tokens_len = len(tok.encode(answer)) + 2  # + [/RESP] + </s>
    assert supervised_tokens == resp_tokens_len

    cfg = CloudyConfig(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=128)
    model = CloudyForCausalLM(cfg)
    model.eval()

    # Format inference prompt
    input_ids = format_inference_prompt(prompt, tok)
    with torch.no_grad():
        logits, _ = model(input_ids)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        top10_probs, top10_ids = torch.topk(probs, 10)

    assert top10_probs.shape == (10,)
    assert top10_ids.shape == (10,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
