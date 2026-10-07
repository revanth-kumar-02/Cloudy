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

    chunks: List[DistillSampleChunk] = []

    # First chunk contains full prompt up to [RESP] and first slice of response
    first_chunk_capacity = max_seq_len - prompt_len
    first_resp_slice = full_resp_ids[:first_chunk_capacity]

    first_input_ids = base_prompt_ids + first_resp_slice
    first_labels = [-100] * len(base_prompt_ids) + first_resp_slice
    first_mask = [1] * len(first_input_ids)

    if pad_to_max_len and len(first_input_ids) < max_seq_len:
        pad_len = max_seq_len - len(first_input_ids)
        first_input_ids += [PAD_TOKEN_ID] * pad_len
        first_labels += [-100] * pad_len
        first_mask += [0] * pad_len

    chunks.append(
        DistillSampleChunk(
            input_ids=torch.tensor(first_input_ids, dtype=torch.long),
            attention_mask=torch.tensor(first_mask, dtype=torch.long),
            labels=torch.tensor(first_labels, dtype=torch.long),
            unpadded_length=len(base_prompt_ids) + len(first_resp_slice),
            example_id=example_id,
            chunk_idx=0,
        )
    )

    # Subsequent continuation chunks: continue response with preceding context, NEVER re-attaching [RESP]
    current_idx = first_chunk_capacity
    while current_idx < len(full_resp_ids):
        # Allow up to half the window for continuation response tokens, remaining for preceding context
        target_slice_len = min(max_seq_len // 2, len(full_resp_ids) - current_idx)
        resp_slice = full_resp_ids[current_idx : current_idx + target_slice_len]

        # Context tokens preceding this slice (BOS + preceding tokens)
        context_cap = max_seq_len - len(resp_slice) - 1  # -1 for BOS
        context_len = min(context_cap, current_idx)
        context_slice = full_resp_ids[current_idx - context_len : current_idx]

        cont_input_ids = [BOS_TOKEN_ID] + context_slice + resp_slice
        # Mask BOS and preceding context tokens so only continuation tokens are supervised
        cont_labels = [-100] * (1 + len(context_slice)) + resp_slice
        cont_mask = [1] * len(cont_input_ids)

        if pad_to_max_len and len(cont_input_ids) < max_seq_len:
            pad_len = max_seq_len - len(cont_input_ids)
            cont_input_ids += [PAD_TOKEN_ID] * pad_len
            cont_labels += [-100] * pad_len
            cont_mask += [0] * pad_len

        chunks.append(
            DistillSampleChunk(
                input_ids=torch.tensor(cont_input_ids, dtype=torch.long),
                attention_mask=torch.tensor(cont_mask, dtype=torch.long),
                labels=torch.tensor(cont_labels, dtype=torch.long),
                unpadded_length=1 + len(context_slice) + len(resp_slice),
                example_id=example_id,
                chunk_idx=len(chunks),
            )
        )
        current_idx += target_slice_len

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
