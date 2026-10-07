from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

# Cloudy standard special token IDs
BOS_TOKEN_ID = 0      # <s>
EOS_TOKEN_ID = 1      # </s>
PAD_TOKEN_ID = 2      # <pad>
UNK_TOKEN_ID = 3      # <unk>
INST_START_ID = 4     # [INST]
INST_END_ID = 5       # [/INST]
RESP_START_ID = 6     # [RESP]
RESP_END_ID = 7       # [/RESP]


@dataclass
class TokenizerProtocol:
    """Minimal interface for tokenizer encoding."""
    def encode(self, text: str) -> Any: ...


def prepare_distill_example(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_seq_len: int = 128,
    pad_to_max_len: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Prepare supervised training chunks for an instruction-response pair.
    
    Format:
      <s>[INST] {prompt} [/INST][RESP] {response} [/RESP]</s>

    Supervision:
      - Prompt tokens, special markers, and padding are masked with -100 in labels.
      - Response tokens, [/RESP], and </s> are supervised.
      - If prompt + response exceeds max_seq_len, chunks the response across multiple
        sequences with prompt context preserved, ensuring no teacher response tokens
        are silently lost.
    """
    # Tokenize text
    prompt_tokens = tokenizer.encode(prompt).ids if hasattr(tokenizer.encode(prompt), "ids") else tokenizer.encode(prompt)
    resp_tokens = tokenizer.encode(response).ids if hasattr(tokenizer.encode(response), "ids") else tokenizer.encode(response)

    # Base prompt sequence: <s> [INST] prompt [/INST] [RESP]
    base_prompt_ids = [BOS_TOKEN_ID, INST_START_ID] + list(prompt_tokens) + [INST_END_ID, RESP_START_ID]
    # Complete response sequence: response [/RESP] </s>
    full_resp_ids = list(resp_tokens) + [RESP_END_ID, EOS_TOKEN_ID]

    prompt_len = len(base_prompt_ids)
    if prompt_len >= max_seq_len - 2:
        raise ValueError(
            f"Prompt length ({prompt_len}) exceeds max_seq_len ({max_seq_len}) with room for response."
        )

    chunk_capacity = max_seq_len - prompt_len
    chunks: List[Dict[str, torch.Tensor]] = []

    # Slide through response tokens in chunks
    for i in range(0, len(full_resp_ids), chunk_capacity):
        resp_chunk = full_resp_ids[i : i + chunk_capacity]
        seq_input_ids = base_prompt_ids + resp_chunk
        # Prompt tokens are masked with -100
        seq_labels = [-100] * len(base_prompt_ids) + resp_chunk
        seq_attention_mask = [1] * len(seq_input_ids)

        # Padding if requested
        if pad_to_max_len and len(seq_input_ids) < max_seq_len:
            pad_len = max_seq_len - len(seq_input_ids)
            seq_input_ids += [PAD_TOKEN_ID] * pad_len
            seq_labels += [-100] * pad_len
            seq_attention_mask += [0] * pad_len

        chunks.append({
            "input_ids": torch.tensor(seq_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(seq_attention_mask, dtype=torch.long),
            "labels": torch.tensor(seq_labels, dtype=torch.long),
            "unpadded_length": len(base_prompt_ids) + len(resp_chunk),
            "chunk_idx": len(chunks),
        })

    return chunks


class TeacherDistillDataset(Dataset):
    """Dataset for supervised teacher-student distillation examples."""

    def __init__(
        self,
        data_path_or_records: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: Any,
        max_seq_len: int = 128,
        pad_to_max_len: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max_len = pad_to_max_len
        self.raw_records: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, torch.Tensor]] = []

        if isinstance(data_path_or_records, (str, Path)):
            path = Path(data_path_or_records)
            assert path.is_file(), f"Dataset file not found: {path}"
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.raw_records.append(json.loads(line))
        else:
            self.raw_records = list(data_path_or_records)

        self._process_records()

    def _process_records(self):
        for record in self.raw_records:
            messages = record.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

            if not user_msg or not assistant_msg:
                continue

            chunks = prepare_distill_example(
                prompt=user_msg,
                response=assistant_msg,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                pad_to_max_len=self.pad_to_max_len,
            )
            self.samples.extend(chunks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_raw_examples": len(self.raw_records),
            "total_training_chunks": len(self.samples),
            "max_seq_len": self.max_seq_len,
        }
