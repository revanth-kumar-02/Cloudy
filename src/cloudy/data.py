from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from cloudy.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_END_ID,
    INST_START_ID,
    PAD_TOKEN_ID,
    RESP_END_ID,
    RESP_START_ID,
    CloudyTokenizer,
)

logger = logging.getLogger("cloudy.data")


@dataclass
class DistillSampleChunk:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    unpadded_length: int
    example_id: str
    chunk_idx: int


def format_and_chunk_example(
    example_id: str,
    prompt: str,
    response: str,
    tokenizer: CloudyTokenizer,
    max_seq_len: int = 128,
    pad_to_max_len: bool = True,
) -> Tuple[List[DistillSampleChunk], Dict[str, Any]]:
    """
    Formats a user-teacher pair into supervised next-token prediction chunks:
      <s>[INST] {prompt} [/INST][RESP] {response} [/RESP]</s>

    Supervision:
      - Prompt tokens, special markers, and padding are masked with -100 in labels.
      - Assistant response tokens, [/RESP], and </s> are supervised.
      - Preserves full response: if length > max_seq_len, chunks into multiple sequences.
    """
    prompt_ids = tokenizer.encode(prompt)
    resp_ids = tokenizer.encode(response)

    base_prompt_ids = [BOS_TOKEN_ID, INST_START_ID] + list(prompt_ids) + [INST_END_ID, RESP_START_ID]
    full_resp_ids = list(resp_ids) + [RESP_END_ID, EOS_TOKEN_ID]

    total_tokens = len(base_prompt_ids) + len(full_resp_ids)
    fits_single_seq = total_tokens <= max_seq_len

    prompt_len = len(base_prompt_ids)
    if prompt_len >= max_seq_len - 2:
        raise ValueError(
            f"Prompt {example_id} length ({prompt_len}) exceeds available max_seq_len ({max_seq_len})."
        )

    chunk_capacity = max_seq_len - prompt_len
    chunks: List[DistillSampleChunk] = []

    for i in range(0, len(full_resp_ids), chunk_capacity):
        resp_chunk = full_resp_ids[i : i + chunk_capacity]
        seq_input_ids = base_prompt_ids + resp_chunk
        # Prompt tokens are masked with -100
        seq_labels = [-100] * len(base_prompt_ids) + resp_chunk
        seq_attention_mask = [1] * len(seq_input_ids)

        # Apply padding if requested
        if pad_to_max_len and len(seq_input_ids) < max_seq_len:
            pad_len = max_seq_len - len(seq_input_ids)
            seq_input_ids += [PAD_TOKEN_ID] * pad_len
            seq_labels += [-100] * pad_len
            seq_attention_mask += [0] * pad_len

        chunks.append(
            DistillSampleChunk(
                input_ids=torch.tensor(seq_input_ids, dtype=torch.long),
                attention_mask=torch.tensor(seq_attention_mask, dtype=torch.long),
                labels=torch.tensor(seq_labels, dtype=torch.long),
                unpadded_length=len(base_prompt_ids) + len(resp_chunk),
                example_id=example_id,
                chunk_idx=len(chunks),
            )
        )

    stats = {
        "example_id": example_id,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(resp_ids),
        "total_tokens": total_tokens,
        "fits_single_seq": fits_single_seq,
        "num_chunks": len(chunks),
    }
    return chunks, stats


class TeacherDistillationDataset(Dataset):
    """Dataset for supervised teacher-student demonstration examples."""

    def __init__(
        self,
        data_path_or_records: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: CloudyTokenizer,
        max_seq_len: int = 128,
        pad_to_max_len: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max_len = pad_to_max_len

        self.records: List[Dict[str, Any]] = []
        self.chunks: List[DistillSampleChunk] = []
        self.example_stats: List[Dict[str, Any]] = []

        if isinstance(data_path_or_records, (str, Path)):
            p = Path(data_path_or_records)
            if not p.is_file():
                raise FileNotFoundError(f"Teacher dataset not found: {p}")
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        else:
            self.records = list(data_path_or_records)

        self._process_all()

    def _process_all(self):
        for idx, rec in enumerate(self.records):
            rec_id = rec.get("id", f"example_{idx:03d}")
            messages = rec.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

            if not user_msg or not asst_msg:
                continue

            chunks, stats = format_and_chunk_example(
                example_id=rec_id,
                prompt=user_msg,
                response=asst_msg,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                pad_to_max_len=self.pad_to_max_len,
            )
            self.chunks.extend(chunks)
            self.example_stats.append(stats)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        c = self.chunks[idx]
        return {
            "input_ids": c.input_ids,
            "attention_mask": c.attention_mask,
            "labels": c.labels,
        }

    def get_summary(self) -> Dict[str, Any]:
        overflow_count = sum(1 for s in self.example_stats if not s["fits_single_seq"])
        return {
            "total_examples": len(self.records),
            "total_training_chunks": len(self.chunks),
            "max_seq_len": self.max_seq_len,
            "overflow_examples": overflow_count,
            "stats": self.example_stats,
        }


def create_distillation_dataloader(
    dataset: TeacherDistillationDataset,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
