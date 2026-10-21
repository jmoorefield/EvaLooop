"""Claude models through the Anthropic Messages API.

Unlike :class:`~evaloop.models.closed_source.OpenAILLM`, API failures raise exceptions instead of
being returned as response text, so they can never be tested as code. Transient failures (rate
limits, overload, server and connection errors) are retried with exponential backoff first.

The ``anthropic`` package is imported lazily (it is part of the ``evaloop[bigcodebench]`` extra);
tests inject a fake SDK module.
"""

import logging
import os
import random
import time
from typing import Any, Callable, Dict

from .base import BaseLLM

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

# HTTP statuses worth retrying: request timeout, conflict, rate limit, and overloaded (529).
# Every 5xx is retried as well.
_RETRYABLE_STATUSES = frozenset({408, 409, 429, 529})

_INSTALL_HINT = (
    "Claude models need the 'anthropic' package; install it with "
    "pip install -e '.[bigcodebench]' (see the README for the pinned version)."
)


class AnthropicGenerationError(RuntimeError):
    """The Anthropic API failed for a prompt (after any retries)."""


class AnthropicLLM(BaseLLM):
    """Claude via the Anthropic Messages API, with greedy settings and retries.

    Model config keys:
        model_id: Pinned model snapshot, e.g. ``claude-sonnet-4-5-20250929``.
        api_key: API key; falls back to ``$ANTHROPIC_API_KEY``.
        max_tokens: Generation budget per response.
        temperature: Sampling temperature, sent via ``extra_body`` (anthropic 1.x has no
            ``temperature`` argument). Only models that still accept it (the Claude 4.5 line,
            e.g. Sonnet 4.5) can be used. ``top_p`` is never sent.
        system_prompt: System prompt; ``None`` or ``""`` sends none.
        max_retries: Retries of a transient failure (default 6).
        retry_base_delay, retry_max_delay: Backoff bounds in seconds (default 2 and 60).
        request_timeout: Seconds per request (default 600).
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        *,
        sdk: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """
        Args:
            model_config: See the class docstring.
            sdk: Replaces the ``anthropic`` module (tests).
            sleep: Replaces ``time.sleep`` between retries (tests).
        """
        super().__init__(model_config)
        self.model_id = model_config.get("model_id")
        if not self.model_id:
            raise ValueError("AnthropicLLM needs a model_id")
        self.api_key = model_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No Anthropic API key: set ANTHROPIC_API_KEY")
        self.max_tokens = int(model_config.get("max_tokens", 1024))
        self.temperature = float(model_config.get("temperature", 0.0))
        self.system_prompt = model_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.max_retries = int(model_config.get("max_retries", 6))
        self.retry_base_delay = float(model_config.get("retry_base_delay", 2.0))
        self.retry_max_delay = float(model_config.get("retry_max_delay", 60.0))
        self.truncated_responses = 0
        self._sleep = sleep
        self._sdk = sdk if sdk is not None else _import_sdk()
        # The SDK's own retries are disabled so that every retry is ours (and logged).
        self.client = self._sdk.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=float(model_config.get("request_timeout", 600.0)),
        )
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """Returns Claude's response to ``prompt`` as one user message.

        Raises:
            AnthropicGenerationError: If the request fails with a non-retryable error, or a
                transient one persists after ``max_retries`` retries.
        """
        request: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            # anthropic 1.x dropped the sampling parameters from its signatures, not from the API:
            # Claude 4.5-era models such as Sonnet 4.5 still honour temperature, and the protocol
            # needs greedy decoding, so it is sent in the request body directly.
            "extra_body": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        if self.system_prompt:
            request["system"] = self.system_prompt

        attempt = 0
        while True:
            try:
                response = self.client.messages.create(**request)
                break
            except Exception as err:  # classified below; anything else is re-raised as is
                if not isinstance(err, self._sdk.AnthropicError):
                    raise
                if not self._is_retryable(err) or attempt >= self.max_retries:
                    raise AnthropicGenerationError(
                        f"Anthropic API error ({type(err).__name__}) after {attempt} retries: {err}"
                    ) from err
                attempt += 1
                delay = self._backoff(attempt)
                self.logger.warning(
                    "Anthropic API %s, retrying (%d/%d) in %.1fs",
                    type(err).__name__, attempt, self.max_retries, delay,
                )
                self._sleep(delay)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if response.stop_reason == "max_tokens":
            # Returned as-is, as the protocol requires; counted so the run can report it.
            self.truncated_responses += 1
        if not text.strip():
            self.logger.warning("Empty response from %s (stop_reason=%s)", self.model_id, response.stop_reason)
        return text

    def _is_retryable(self, err: Exception) -> bool:
        """Whether a failure is transient: a connection problem, rate limit, overload or 5xx."""
        if isinstance(err, self._sdk.APIConnectionError):
            return True
        status = getattr(err, "status_code", None)
        return isinstance(status, int) and (status in _RETRYABLE_STATUSES or status >= 500)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with equal jitter, capped at ``retry_max_delay``."""
        delay = min(self.retry_max_delay, self.retry_base_delay * 2 ** (attempt - 1))
        return delay / 2 + random.uniform(0, delay / 2)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name}, model_id={self.model_id})"


def _import_sdk() -> Any:
    """Imports the ``anthropic`` package, with an install hint if it is missing."""
    try:
        import anthropic
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    return anthropic
