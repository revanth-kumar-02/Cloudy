from pathlib import Path
import pytest
import torch

from cloudy.data import (
    TeacherDistillationDataset,
    format_and_chunk_example,
)
from cloudy.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_END_ID,
    INST_START_ID,
    PAD_TOKEN_ID,
    RESP_END_ID,
    RESP_START_ID,
    MockCloudyTokenizer,
)


def test_special_tokens_definitions():
    assert BOS_TOKEN_ID == 0
    assert EOS_TOKEN_ID == 1
    assert PAD_TOKEN_ID == 2
    assert INST_START_ID == 4
    assert INST_END_ID == 5
    assert RESP_START_ID == 6
    assert RESP_END_ID == 7


def test_tokenizer_encoding_decoding():
    tok = MockCloudyTokenizer(vocab_size=1000)
    text = "Hello world from Cloudy"
    ids = tok.encode(text)
    assert len(ids) == 4
    decoded = tok.decode(ids)
    assert "tok_" in decoded
    assert tok.get_vocab_size() == 1000


def test_format_and_chunk_example_prompt_masking_and_padding():
    tok = MockCloudyTokenizer(vocab_size=1000)
    prompt = "What is a function?"
    response = "A function is reusable code."

    chunks, stats = format_and_chunk_example(
        example_id="ex_001",
        prompt=prompt,
        response=response,
        tokenizer=tok,
        max_seq_len=64,
        pad_to_max_len=True,
    )

    assert len(chunks) == 1
    chunk = chunks[0]

    # Verify tensor shapes
    assert chunk.input_ids.shape == (64,)
    assert chunk.labels.shape == (64,)
    assert chunk.attention_mask.shape == (64,)

    # Starts with <s> (0) and [INST] (4)
    assert chunk.input_ids[0].item() == BOS_TOKEN_ID
    assert chunk.input_ids[1].item() == INST_START_ID

    # Find [RESP] position
    resp_start_idx = (chunk.input_ids == RESP_START_ID).nonzero(as_tuple=True)[0].item()

    # All tokens up to and including [RESP] MUST be masked with -100
    assert (chunk.labels[: resp_start_idx + 1] == -100).all()

    # Tokens immediately following [RESP] must NOT be -100 (they are supervised)
    assert chunk.labels[resp_start_idx + 1].item() != -100

    # Padding tokens must have attention_mask=0 and label=-100
    unpadded_len = chunk.unpadded_length
    assert (chunk.labels[unpadded_len:] == -100).all()
    assert (chunk.attention_mask[unpadded_len:] == 0).all()


def test_long_sequence_non_truncating_chunking():
    tok = MockCloudyTokenizer(vocab_size=1000)
    prompt = "Explain lists"
    # Long response with 80 words
    response = " ".join([f"item_{i}" for i in range(80)])

    chunks, stats = format_and_chunk_example(
        example_id="ex_long",
        prompt=prompt,
        response=response,
        tokenizer=tok,
        max_seq_len=64,
        pad_to_max_len=True,
    )

    # Must produce multiple chunks because 80 words > 64 max_seq_len
    assert len(chunks) > 1
    assert stats["fits_single_seq"] is False

    # Verify no response tokens are dropped
    original_resp_tokens = len(tok.encode(response)) + 2  # + [/RESP] + </s>
    collected_tokens = []
    for c in chunks:
        unmasked = c.labels[c.labels != -100].tolist()
        collected_tokens.extend(unmasked)

    assert len(collected_tokens) == original_resp_tokens


def test_teacher_distillation_dataset_interface():
    tok = MockCloudyTokenizer(vocab_size=1000)
    records = [
        {
            "id": "q1",
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A high-level programming language."},
            ],
        },
        {
            "id": "q2",
            "messages": [
                {"role": "user", "content": "What is an SLM?"},
                {"role": "assistant", "content": "A small language model."},
            ],
        },
    ]

    dataset = TeacherDistillationDataset(records, tokenizer=tok, max_seq_len=64)
    assert len(dataset) >= 2
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert dataset.get_summary()["total_examples"] == 2
