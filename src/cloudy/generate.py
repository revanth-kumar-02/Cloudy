from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from cloudy.model import CloudyForCausalLM
from cloudy.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    INST_END_ID,
    INST_START_ID,
    RESP_END_ID,
    RESP_START_ID,
    CloudyTokenizer,
)

logger = logging.getLogger("cloudy.generate")


def format_inference_prompt(prompt: str, tokenizer: CloudyTokenizer) -> torch.Tensor:
    """Formats raw prompt into model input: <s>[INST] {prompt} [/INST][RESP]"""
    prompt_tokens = tokenizer.encode(prompt)
    seq = [BOS_TOKEN_ID, INST_START_ID] + list(prompt_tokens) + [INST_END_ID, RESP_START_ID]
    return torch.tensor([seq], dtype=torch.long)


@torch.no_grad()
def generate_response(
    model: CloudyForCausalLM,
    tokenizer: CloudyTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 20,
    device: Optional[torch.device] = None,
) -> str:
    """Generates assistant response for a given instruction prompt."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = format_inference_prompt(prompt, tokenizer).to(device)
    prompt_len = input_ids.shape[1]

    curr_ids = input_ids.clone()
    stop_tokens = {RESP_END_ID, EOS_TOKEN_ID}

    for _ in range(max_new_tokens):
        if curr_ids.shape[1] >= model.config.max_seq_len:
            break

        logits, _ = model(curr_ids)
        next_logits = logits[:, -1, :]

        if temperature <= 0.0 or temperature < 1e-4:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled = next_logits / temperature
            if top_k > 0:
                top_k_val = min(top_k, scaled.size(-1))
                val, _ = torch.topk(scaled, top_k_val)
                scaled[scaled < val[..., -1, None]] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id in stop_tokens:
            break

        curr_ids = torch.cat([curr_ids, next_token], dim=1)

    # Extract response portion
    generated_tokens = curr_ids[0, prompt_len:].tolist()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
