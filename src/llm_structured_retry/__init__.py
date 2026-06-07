"""llm-structured-retry: retry LLM calls on validation failure.

When an LLM returns output that fails validation, inject the validation error
as a follow-up human message and retry — up to N times total. Unlike local
JSON-repair approaches, this sends the error back to the LLM and asks it to
fix its own output.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


class ValidationError(Exception):
    """Raised by a validate_fn to signal that LLM output is invalid."""


class MaxRetriesExceededError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Max retries ({attempts}) exceeded. Last error: {last_error}")


class RetryResult:
    """Result of a retry_call / retry_call_async invocation."""

    def __init__(self, content: str, attempts: int, succeeded: bool, errors: list[str]) -> None:
        self.content = content
        self.attempts = attempts
        self.succeeded = succeeded
        # One entry per failed attempt (length == attempts - 1 on success,
        # length == max_retries on total failure before raising).
        self.errors: list[str] = errors

    def __repr__(self) -> str:
        return (
            f"RetryResult(attempts={self.attempts}, succeeded={self.succeeded}, "
            f"errors={self.errors!r}, content={self.content[:40]!r})"
        )


_DEFAULT_ERROR_TEMPLATE = "Your last response had an error: {error}. Please try again."


class StructuredRetry:
    """Retry an LLM call when its output fails validation.

    Parameters
    ----------
    max_retries:
        Total number of attempts allowed (first attempt + subsequent retries).
        Minimum is 1. A value of 1 means exactly one attempt; if it fails,
        MaxRetriesExceededError is raised immediately.
    validate_fn:
        Callable[[str], None].  Should raise ValidationError with a message
        when the LLM output is invalid, or return None when it is acceptable.
        If None, every response is treated as valid.
    error_template:
        Format string injected back into the conversation on each failure.
        Must contain ``{error}`` placeholder.
    """

    def __init__(
        self,
        max_retries: int = 3,
        validate_fn: Callable[[str], None] | None = None,
        error_template: str | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.max_retries = max_retries
        self.validate_fn = validate_fn
        self.error_template = error_template or _DEFAULT_ERROR_TEMPLATE

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def retry_call(
        self,
        llm_fn: Callable[..., str],
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> RetryResult:
        """Call llm_fn(messages, **kwargs) -> str, retrying on ValidationError.

        The original ``messages`` list is never mutated; a copy is made for
        each attempt so callers retain their original history unchanged.
        """
        errors: list[str] = []
        # Working copy — each retry appends to this copy, not the caller's list.
        current_messages = list(messages)

        for attempt in range(1, self.max_retries + 1):
            content = llm_fn(current_messages, **kwargs)

            if self.validate_fn is None:
                return RetryResult(content=content, attempts=attempt, succeeded=True, errors=errors)

            try:
                self.validate_fn(content)
                return RetryResult(content=content, attempts=attempt, succeeded=True, errors=errors)
            except ValidationError as exc:
                error_msg = str(exc)
                errors.append(error_msg)

                if attempt < self.max_retries:
                    # Inject the error as a follow-up user message for the next attempt.
                    # Append the assistant turn first so conversation history is coherent.
                    current_messages = current_messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": self.error_template.format(error=error_msg),
                        },
                    ]

        raise MaxRetriesExceededError(attempts=self.max_retries, last_error=errors[-1])

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def retry_call_async(
        self,
        llm_fn: Callable[..., Any],  # async (messages, **kwargs) -> str
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> RetryResult:
        """Async variant of retry_call. llm_fn must be an async callable."""
        errors: list[str] = []
        current_messages = list(messages)

        for attempt in range(1, self.max_retries + 1):
            content = await llm_fn(current_messages, **kwargs)

            if self.validate_fn is None:
                return RetryResult(content=content, attempts=attempt, succeeded=True, errors=errors)

            try:
                self.validate_fn(content)
                return RetryResult(content=content, attempts=attempt, succeeded=True, errors=errors)
            except ValidationError as exc:
                error_msg = str(exc)
                errors.append(error_msg)

                if attempt < self.max_retries:
                    current_messages = current_messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": self.error_template.format(error=error_msg),
                        },
                    ]

        raise MaxRetriesExceededError(attempts=self.max_retries, last_error=errors[-1])

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    def validated(self, validate_fn: Callable[[str], None]) -> Callable:
        """Decorator — wraps a sync llm_fn so it retries with validate_fn.

        Usage::

            retry = StructuredRetry(max_retries=3)

            @retry.validated(my_validator)
            def call_llm(messages):
                ...

            result = call_llm(messages)  # returns RetryResult
        """

        def decorator(llm_fn: Callable[..., str]) -> Callable:
            @functools.wraps(llm_fn)
            def wrapper(messages: list[dict[str, Any]], **kwargs: Any) -> RetryResult:
                sr = StructuredRetry(
                    max_retries=self.max_retries,
                    validate_fn=validate_fn,
                    error_template=self.error_template,
                )
                return sr.retry_call(llm_fn, messages, **kwargs)

            return wrapper

        return decorator

    def validated_async(self, validate_fn: Callable[[str], None]) -> Callable:
        """Decorator — wraps an async llm_fn so it retries with validate_fn.

        Usage::

            retry = StructuredRetry(max_retries=3)

            @retry.validated_async(my_validator)
            async def call_llm(messages):
                ...

            result = await call_llm(messages)  # returns RetryResult
        """

        def decorator(llm_fn: Callable[..., Any]) -> Callable:
            @functools.wraps(llm_fn)
            async def wrapper(messages: list[dict[str, Any]], **kwargs: Any) -> RetryResult:
                sr = StructuredRetry(
                    max_retries=self.max_retries,
                    validate_fn=validate_fn,
                    error_template=self.error_template,
                )
                return await sr.retry_call_async(llm_fn, messages, **kwargs)

            return wrapper

        return decorator


__all__ = [
    "ValidationError",
    "MaxRetriesExceededError",
    "RetryResult",
    "StructuredRetry",
]
