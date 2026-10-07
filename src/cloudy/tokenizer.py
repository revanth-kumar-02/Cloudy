from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

# Cloudy Standard Special Tokens
BOS_TOKEN_ID: int = 0      # <s>
EOS_TOKEN_ID: int = 1      # </s>
PAD_TOKEN_ID: int = 2      # <pad>
UNK_TOKEN_ID: int = 3      # <unk>
INST_START_ID: int = 4     # [INST]
INST_END_ID: int = 5       # [/INST]
RESP_START_ID: int = 6     # [RESP]
RESP_END_ID: int = 7       # [/RESP]

SPECIAL_TOKEN_MAP = {
    "<s>": BOS_TOKEN_ID,
    "</s>": EOS_TOKEN_ID,
    "<pad>": PAD_TOKEN_ID,
    "<unk>": UNK_TOKEN_ID,
    "[INST]": INST_START_ID,
    "[/INST]": INST_END_ID,
    "[RESP]": RESP_START_ID,
    "[/RESP]": RESP_END_ID,
}


class CloudyTokenizer:
    """Tokenizer adapter wrapping the Byte-Level BPE tokenizer with Cloudy special tokens."""

    def __init__(self, tokenizer_backend: Any):
        self._tokenizer = tokenizer_backend
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.unk_token_id = UNK_TOKEN_ID
        self.inst_start_id = INST_START_ID
        self.inst_end_id = INST_END_ID
        self.resp_start_id = RESP_START_ID
        self.resp_end_id = RESP_END_ID

    @classmethod
    def from_file(cls, path_or_dir: Union[str, Path]) -> CloudyTokenizer:
        from tokenizers import Tokenizer

        p = Path(path_or_dir)
        if p.is_dir():
            json_path = p / "tokenizer.json"
        else:
            json_path = p

        if not json_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {json_path}")

        raw_tokenizer = Tokenizer.from_file(str(json_path))
        return cls(raw_tokenizer)

    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of token IDs."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids if hasattr(encoding, "ids") else list(encoding)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a string."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Returns vocabulary size."""
        if hasattr(self._tokenizer, "get_vocab_size"):
            return self._tokenizer.get_vocab_size()
        return len(self._tokenizer)

    def __len__(self) -> int:
        return self.get_vocab_size()


class MockCloudyTokenizer(CloudyTokenizer):
    """Deterministic in-memory mock tokenizer for unit tests without external files."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        super().__init__(self)

    def encode(self, text: str) -> List[int]:
        # Hash each word into an ID outside the special tokens range (8 .. vocab_size-1)
        words = text.split()
        if not words:
            return []
        tokens = []
        for w in words:
            token_id = 8 + (abs(hash(w)) % (self.vocab_size - 8))
            tokens.append(token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in SPECIAL_TOKEN_MAP.values():
                continue
            words.append(f"tok_{tid}")
        return " ".join(words)

    def get_vocab_size(self) -> int:
        return self.vocab_size
