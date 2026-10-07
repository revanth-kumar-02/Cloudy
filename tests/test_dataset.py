from types import SimpleNamespace
import pytest
import torch

from cloudy.dataset import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    INST_START_ID,
    INST_END_ID,
    RESP_START_ID,
    RESP_END_ID,
    TeacherDistillDataset,
    prepare_distill_example,
)


class MockTokenizer:
    """Mock tokenizer for offline unit tests."""
    def encode(self, text: str):
        # Convert words to deterministic pseudo-token IDs
        tokens = [(hash(word) % 30000) + 10 for word in text.split()]
        return SimpleNamespace(ids=tokens)


def test_special_token_ids():
    assert BOS_TOKEN_ID == 0
    assert EOS_TOKEN_ID == 1
    assert PAD_TOKEN_ID == 2
    assert INST_START_ID == 4
    assert INST_END_ID == 5
    assert RESP_START_ID == 6
    assert RESP_END_ID == 7


def test_prepare_distill_example_basic():
    tokenizer = MockTokenizer()
    prompt = "What is a variable?"
    response = "A variable is a labeled container."

    chunks = prepare_distill_example(
        prompt=prompt,
        response=response,
        tokenizer=tokenizer,
        max_seq_len=128,
        pad_to_max_len=True,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    input_ids = chunk["input_ids"]
    labels = chunk["labels"]
    attention_mask = chunk["attention_mask"]

    assert len(input_ids) == 128
    assert len(labels) == 128
    assert len(attention_mask) == 128

    # First token is <s> (0), then [INST] (4)
    assert input_ids[0].item() == 0
    assert input_ids[1].item() == 4

    # Labels for prompt and special tokens up to [RESP] (6) must be -100
    prompt_len = len(tokenizer.encode(prompt).ids)
    # Indices: 0 (<s>), 1 ([INST]), 2..1+prompt_len (prompt), 2+prompt_len ([/INST]), 3+prompt_len ([RESP])
    resp_start_idx = 3 + prompt_len
    assert input_ids[resp_start_idx].item() == RESP_START_ID
    assert (labels[: resp_start_idx + 1] == -100).all()

    # Response tokens must not be -100
    resp_token_idx = resp_start_idx + 1
    assert labels[resp_token_idx].item() != -100

    # End of response has [/RESP] (7) and </s> (1)
    unpadded_len = chunk["unpadded_length"]
    assert input_ids[unpadded_len - 2].item() == RESP_END_ID
    assert input_ids[unpadded_len - 1].item() == EOS_TOKEN_ID

    # Padding tokens must have label=-100 and attention_mask=0
    assert (labels[unpadded_len:] == -100).all()
    assert (attention_mask[unpadded_len:] == 0).all()


def test_chunking_preserves_overflow_tokens():
    tokenizer = MockTokenizer()
    prompt = "Explain functions"
    # Long response with 150 words
    response = " ".join([f"word{i}" for i in range(150)])

    chunks = prepare_distill_example(
        prompt=prompt,
        response=response,
        tokenizer=tokenizer,
        max_seq_len=128,
        pad_to_max_len=True,
    )

    # Must be split into multiple chunks without truncating
    assert len(chunks) > 1

    # Total supervised response tokens across chunks must match original response + [/RESP] + </s>
    original_resp_len = len(tokenizer.encode(response).ids) + 2  # + [/RESP] + </s>
    collected_resp_tokens = []
    for chunk in chunks:
        # Get unmasked tokens (not -100)
        unmasked = chunk["labels"][chunk["labels"] != -100].tolist()
        collected_resp_tokens.extend(unmasked)

    assert len(collected_resp_tokens) == original_resp_len


def test_teacher_distill_dataset():
    tokenizer = MockTokenizer()
    records = [
        {
            "id": "test_1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
        {
            "id": "test_2",
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A high-level programming language."},
            ],
        },
    ]

    dataset = TeacherDistillDataset(records, tokenizer=tokenizer, max_seq_len=64)
    assert len(dataset) >= 2
    sample = dataset[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert "attention_mask" in sample
    assert dataset.stats()["total_raw_examples"] == 2
