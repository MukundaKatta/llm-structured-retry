"""Tests for llm_structured_retry."""

from __future__ import annotations

import pytest

from llm_structured_retry import (
    MaxRetriesExceededError,
    RetryResult,
    StructuredRetry,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "Say hello."}]


def make_llm(responses: list[str]):
    """Sync llm_fn that returns successive responses."""
    it = iter(responses)

    def llm_fn(messages, **kwargs):
        return next(it)

    return llm_fn


async def make_llm_async(responses: list[str]):
    """Return an async llm_fn that returns successive responses."""
    responses_list = list(responses)
    state = {"idx": 0}

    async def llm_fn(messages, **kwargs):
        val = responses_list[state["idx"]]
        state["idx"] += 1
        return val

    return llm_fn


def require_json(content: str) -> None:
    """Validate that content is JSON."""
    import json

    try:
        json.loads(content)
    except Exception as exc:
        raise ValidationError(f"not valid JSON: {exc}") from exc


def require_hello(content: str) -> None:
    """Validate that content contains 'hello'."""
    if "hello" not in content.lower():
        raise ValidationError("response must contain 'hello'")


# ---------------------------------------------------------------------------
# 1. No validate_fn — always succeeds on first try
# ---------------------------------------------------------------------------


def test_no_validate_fn_succeeds_first_try():
    sr = StructuredRetry(max_retries=3)
    result = sr.retry_call(make_llm(["ok"]), list(MESSAGES))
    assert isinstance(result, RetryResult)
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.content == "ok"
    assert result.errors == []


# ---------------------------------------------------------------------------
# 2. validate_fn passes on first try
# ---------------------------------------------------------------------------


def test_validate_fn_passes_first_try():
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    result = sr.retry_call(make_llm(["Hello there"]), list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.errors == []


# ---------------------------------------------------------------------------
# 3. validate_fn fails once, then passes
# ---------------------------------------------------------------------------


def test_validate_fn_fails_once_then_passes():
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    result = sr.retry_call(make_llm(["bad response", "hello world"]), list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 2
    assert len(result.errors) == 1
    assert "hello" in result.errors[0]


# ---------------------------------------------------------------------------
# 4. validate_fn always fails — raises MaxRetriesExceededError
# ---------------------------------------------------------------------------


def test_always_fails_raises_max_retries():
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        sr.retry_call(make_llm(["bad", "bad", "bad"]), list(MESSAGES))
    err = exc_info.value
    assert err.attempts == 3
    assert "hello" in err.last_error


# ---------------------------------------------------------------------------
# 5. Custom error_template is used
# ---------------------------------------------------------------------------


def test_custom_error_template_forwarded():
    """Custom template appears in the injected follow-up message."""
    captured_messages: list[list[dict]] = []

    def llm_fn(messages, **kwargs):
        captured_messages.append(list(messages))
        # Fail on first call, succeed on second.
        if len(captured_messages) == 1:
            return "bad"
        return "hello"

    sr = StructuredRetry(
        max_retries=3,
        validate_fn=require_hello,
        error_template="CUSTOM: {error} — please fix.",
    )
    result = sr.retry_call(llm_fn, list(MESSAGES))
    assert result.succeeded is True
    # Second call's messages should include the custom template text.
    second_call_messages = captured_messages[1]
    user_followup = next(
        m for m in second_call_messages if m["role"] == "user" and "CUSTOM:" in m["content"]
    )
    assert "CUSTOM:" in user_followup["content"]
    assert "please fix." in user_followup["content"]


# ---------------------------------------------------------------------------
# 6. Original messages list is NOT mutated
# ---------------------------------------------------------------------------


def test_original_messages_not_mutated():
    original = [{"role": "user", "content": "test"}]
    snapshot = [dict(m) for m in original]
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    # Fails twice then succeeds.
    sr.retry_call(make_llm(["bad", "bad", "hello"]), original)
    assert original == snapshot, "original messages list must not be mutated"


# ---------------------------------------------------------------------------
# 7. RetryResult.errors length matches number of failed attempts
# ---------------------------------------------------------------------------


def test_errors_length_matches_failed_attempts():
    sr = StructuredRetry(max_retries=5, validate_fn=require_hello)
    result = sr.retry_call(make_llm(["x", "x", "x", "hello"]), list(MESSAGES))
    assert result.attempts == 4
    assert len(result.errors) == 3  # three failures before success


def test_errors_length_on_total_failure():
    sr = StructuredRetry(max_retries=4, validate_fn=require_hello)
    with pytest.raises(MaxRetriesExceededError):
        sr.retry_call(make_llm(["x", "x", "x", "x"]), list(MESSAGES))
    # We can't inspect RetryResult here, but MaxRetriesExceededError.attempts covers it.


# ---------------------------------------------------------------------------
# 8. max_retries=1 — one attempt total, fails immediately
# ---------------------------------------------------------------------------


def test_max_retries_one_fails_immediately():
    sr = StructuredRetry(max_retries=1, validate_fn=require_hello)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        sr.retry_call(make_llm(["bad"]), list(MESSAGES))
    assert exc_info.value.attempts == 1


def test_max_retries_one_succeeds():
    sr = StructuredRetry(max_retries=1, validate_fn=require_hello)
    result = sr.retry_call(make_llm(["hello"]), list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 1


# ---------------------------------------------------------------------------
# 9. Async variant — basic success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_no_validate_fn():
    llm_fn = await make_llm_async(["async ok"])
    sr = StructuredRetry(max_retries=3)
    result = await sr.retry_call_async(llm_fn, list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.content == "async ok"


@pytest.mark.asyncio
async def test_async_validate_passes_first_try():
    llm_fn = await make_llm_async(["hello async"])
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    result = await sr.retry_call_async(llm_fn, list(MESSAGES))
    assert result.attempts == 1
    assert result.succeeded is True


@pytest.mark.asyncio
async def test_async_validate_fails_once_then_passes():
    llm_fn = await make_llm_async(["bad", "hello"])
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    result = await sr.retry_call_async(llm_fn, list(MESSAGES))
    assert result.attempts == 2
    assert result.succeeded is True
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_async_always_fails_raises():
    llm_fn = await make_llm_async(["x", "x", "x"])
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await sr.retry_call_async(llm_fn, list(MESSAGES))
    assert exc_info.value.attempts == 3


@pytest.mark.asyncio
async def test_async_original_messages_not_mutated():
    original = [{"role": "user", "content": "async test"}]
    snapshot = [dict(m) for m in original]
    llm_fn = await make_llm_async(["bad", "hello"])
    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    await sr.retry_call_async(llm_fn, original)
    assert original == snapshot


@pytest.mark.asyncio
async def test_async_max_retries_one():
    llm_fn = await make_llm_async(["bad"])
    sr = StructuredRetry(max_retries=1, validate_fn=require_hello)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await sr.retry_call_async(llm_fn, list(MESSAGES))
    assert exc_info.value.attempts == 1


# ---------------------------------------------------------------------------
# 10. validated() decorator — sync
# ---------------------------------------------------------------------------


def test_validated_decorator_wraps_correctly():
    retry = StructuredRetry(max_retries=3)
    call_count = {"n": 0}

    @retry.validated(require_hello)
    def call_llm(messages, **kwargs):
        call_count["n"] += 1
        return "hello" if call_count["n"] >= 2 else "bad"

    result = call_llm(list(MESSAGES))
    assert isinstance(result, RetryResult)
    assert result.succeeded is True
    assert result.attempts == 2


def test_validated_decorator_raises_when_all_fail():
    retry = StructuredRetry(max_retries=2)

    @retry.validated(require_hello)
    def call_llm(messages, **kwargs):
        return "bad"

    with pytest.raises(MaxRetriesExceededError):
        call_llm(list(MESSAGES))


# ---------------------------------------------------------------------------
# 11. validated_async() decorator — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validated_async_decorator_wraps_correctly():
    retry = StructuredRetry(max_retries=3)
    state = {"n": 0}

    @retry.validated_async(require_hello)
    async def call_llm(messages, **kwargs):
        state["n"] += 1
        return "hello" if state["n"] >= 2 else "nope"

    result = await call_llm(list(MESSAGES))
    assert isinstance(result, RetryResult)
    assert result.succeeded is True
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_validated_async_decorator_raises():
    retry = StructuredRetry(max_retries=2)

    @retry.validated_async(require_hello)
    async def call_llm(messages, **kwargs):
        return "wrong"

    with pytest.raises(MaxRetriesExceededError):
        await call_llm(list(MESSAGES))


# ---------------------------------------------------------------------------
# 12. MaxRetriesExceededError attributes
# ---------------------------------------------------------------------------


def test_max_retries_exceeded_error_attributes():
    err = MaxRetriesExceededError(attempts=4, last_error="it broke")
    assert err.attempts == 4
    assert err.last_error == "it broke"
    assert "4" in str(err)
    assert "it broke" in str(err)


def test_max_retries_exceeded_is_exception():
    with pytest.raises(MaxRetriesExceededError):
        raise MaxRetriesExceededError(attempts=1, last_error="oops")


# ---------------------------------------------------------------------------
# 13. ValidationError is an Exception
# ---------------------------------------------------------------------------


def test_validation_error_is_exception():
    with pytest.raises(ValidationError):
        raise ValidationError("bad output")


# ---------------------------------------------------------------------------
# 14. invalid max_retries
# ---------------------------------------------------------------------------


def test_invalid_max_retries_raises():
    with pytest.raises(ValueError):
        StructuredRetry(max_retries=0)


# ---------------------------------------------------------------------------
# 15. kwargs forwarded to llm_fn
# ---------------------------------------------------------------------------


def test_kwargs_forwarded():
    received_kwargs: list[dict] = []

    def llm_fn(messages, **kwargs):
        received_kwargs.append(kwargs)
        return "hello"

    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    sr.retry_call(llm_fn, list(MESSAGES), temperature=0.7, model="gpt-5")
    assert received_kwargs[0] == {"temperature": 0.7, "model": "gpt-5"}


# ---------------------------------------------------------------------------
# 16. RetryResult repr is human-readable
# ---------------------------------------------------------------------------


def test_retry_result_repr():
    r = RetryResult(content="hello world", attempts=2, succeeded=True, errors=["bad"])
    text = repr(r)
    assert "attempts=2" in text
    assert "succeeded=True" in text


# ---------------------------------------------------------------------------
# 17. default error_template is used when none provided
# ---------------------------------------------------------------------------


def test_default_error_template_used():
    captured: list[str] = []

    def llm_fn(messages, **kwargs):
        # Record the last user message content each call
        for m in reversed(messages):
            if m["role"] == "user":
                captured.append(m["content"])
                break
        return "hello" if len(captured) >= 2 else "bad"

    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    sr.retry_call(llm_fn, list(MESSAGES))
    # Second invocation's last user message should be the default template
    assert "Your last response had an error" in captured[1]


# ---------------------------------------------------------------------------
# 18. Async kwargs forwarded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_kwargs_forwarded():
    received_kwargs: list[dict] = []

    async def llm_fn(messages, **kwargs):
        received_kwargs.append(kwargs)
        return "hello"

    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    await sr.retry_call_async(llm_fn, list(MESSAGES), stream=False, max_tokens=100)
    assert received_kwargs[0] == {"stream": False, "max_tokens": 100}


# ---------------------------------------------------------------------------
# 19. JSON validation — the headline use case (bad JSON repaired on retry)
# ---------------------------------------------------------------------------


def test_require_json_repairs_on_retry():
    sr = StructuredRetry(max_retries=3, validate_fn=require_json)
    result = sr.retry_call(make_llm(["not json", '{"name": "ok"}']), list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 2
    assert len(result.errors) == 1
    assert "not valid JSON" in result.errors[0]
    assert result.content == '{"name": "ok"}'


def test_require_json_first_try_valid():
    sr = StructuredRetry(max_retries=2, validate_fn=require_json)
    result = sr.retry_call(make_llm(['{"a": 1}']), list(MESSAGES))
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.errors == []


# ---------------------------------------------------------------------------
# 20. The invalid assistant reply is injected into the next attempt's history
# ---------------------------------------------------------------------------


def test_invalid_assistant_reply_injected_into_history():
    captured: list[list[dict]] = []

    def llm_fn(messages, **kwargs):
        captured.append([dict(m) for m in messages])
        return "bad reply" if len(captured) == 1 else "hello"

    sr = StructuredRetry(max_retries=3, validate_fn=require_hello)
    result = sr.retry_call(llm_fn, list(MESSAGES))
    assert result.succeeded is True
    # The second call must contain the first (invalid) reply as an assistant turn.
    second_call = captured[1]
    assert {"role": "assistant", "content": "bad reply"} in second_call
