from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cloudy.distillation")


class TeacherClient(ABC):
    """Abstract interface for teacher model providers."""

    @abstractmethod
    def generate_demonstration(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a teacher response for a prompt."""
        pass


class GroqTeacherClient(TeacherClient):
    """Teacher client using Groq's OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self._api_key = api_key or self._resolve_api_key()

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY could not be resolved from environment or Colab secrets. "
                "Ensure GROQ_API_KEY is configured in Colab Secrets or as an environment variable."
            )

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self._api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=self.timeout,
            max_retries=2,
        )
        logger.info(f"Initialized Groq teacher client with model: {self.model_name}")

    def _resolve_api_key(self) -> Optional[str]:
        # Check environment variable first
        key = os.environ.get("GROQ_API_KEY")
        if key:
            return key

        # Check Google Colab secrets if available
        try:
            from google.colab import userdata
            return userdata.get("GROQ_API_KEY")
        except Exception:
            pass

        return None

    def generate_demonstration(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 250,
    ) -> str:
        sys_content = system_prompt or (
            "You are a helpful AI assistant. Give accurate, clear, self-contained answers "
            "suitable for a beginner. Do not invent facts."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError(f"Teacher {self.model_name} returned an empty response.")
        return answer
